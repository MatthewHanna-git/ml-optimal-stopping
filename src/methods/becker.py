"""Deep optimal stopping (DOS), Becker, Cheridito and Jentzen (2019), in its
recursive form.

One stopping network per exercise date, trained backwards from maturity.
At date n, with the later networks trained and fixed, maximise

    E[ g_n(X_n) F_n(X_n) + G_{n+1} (1 - F_n(X_n)) ]        (paper eq. 15)

where F_n = sigmoid(logit_n) is the soft stopping probability at date n and
G_{n+1} is the discounted payoff of executing the hard decisions
(logit >= 0, paper eq. 8) from date n + 1 along the same path. Payoffs are
discounted to time 0, and the decision at maturity is to collect if in the
money.

Departures from the paper:
- The shared 2 x 50 trunk (StopNet, with batch normalisation as in the
  paper) and Adam at Params.lr, cosine-decayed per date, in place of the
  paper's d + 40 widths and learning-rate schedule.
- Each date's network starts from the next date's trained weights. This
  changes the initialisation only, not the objective.
- Params.n_iters is the total budget, split evenly across the N - 1
  trainable dates.
- The g > 0 guard applies to the hard decisions in training and execution
  alike, so the trained rule is the executed one.

fit() returns a BeckerPolicy for certification; price() values the executed
policy on fresh paths.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params
from .common import StopNet


class BeckerPolicy:
    """Hard stopping rule from trained per-date logit networks."""

    def __init__(self, params: Params, payoff_fn, nets):
        self.K = params.K_strike
        self.n_steps = params.n_steps
        self.payoff_fn = payoff_fn
        self.nets = nets                        # {k: StopNet}, k = 1..N-1

    @torch.no_grad()
    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        return (self.nets[k](S) >= 0.0) & (g > 0.0)


def _tail_value(policy, paths, k_from, params, payoff_fn, disc):
    """Time-0-discounted payoff of executing the hard policy from date
    k_from+1 onward along given full paths: (batch,). No gradients."""
    with torch.no_grad():
        N = params.n_steps
        batch = paths.shape[0]
        payoff = torch.zeros(batch, device=paths.device)
        alive = torch.ones(batch, dtype=torch.bool, device=paths.device)
        for m in range(k_from + 1, N + 1):
            ex = policy.exercise(m, paths[:, m, :])
            stop = alive & ex
            payoff[stop] = disc[m] * payoff_fn(paths[stop, m, :], params.K_strike)
            alive &= ~ex
        return payoff


def fit(S0, params: Params, payoff_fn=None, device="cpu", generator=None,
        lr_schedule="cosine"):
    """Backward-recursive DOS training. Returns (BeckerPolicy, diagnostics).

    lr_schedule: "cosine" (Params.lr, cosine-decayed per date) or "paper"
    (the paper's 0.05 -> 0.005 -> 0.0005 at thirds of each date's budget).
    """
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    from ..simulate import gbm_paths

    N = params.n_steps
    dt = params.T / N
    disc = [math.exp(-params.r * k * dt) for k in range(N + 1)]
    if generator is None:
        seed = params.seed if params.seed is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed)

    steps_per_date = max(1, math.ceil(params.n_iters / max(N - 1, 1)))
    nets = {}
    policy = BeckerPolicy(params, payoff_fn, nets)
    final_obj = {}
    prev_state = None

    t0 = time.perf_counter()
    for n in range(N - 1, 0, -1):
        net = StopNet(params.dim, params.arch, params.K_strike,
                      payoff_fn).to(device)
        if prev_state is not None:
            net.load_state_dict(prev_state)     # warm start from date n+1
        net.train()
        if lr_schedule == "paper":
            opt = torch.optim.Adam(net.parameters(), lr=0.05)
            sched = torch.optim.lr_scheduler.MultiStepLR(
                opt, [steps_per_date // 3, (2 * steps_per_date) // 3], gamma=0.1)
        else:
            opt = torch.optim.Adam(net.parameters(), lr=params.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_per_date)
        for _ in range(steps_per_date):
            paths = gbm_paths(S0, params, n_paths=params.batch_size,
                              device=device, generator=generator)
            S_n = paths[:, n, :]
            g_n = payoff_fn(S_n, params.K_strike)
            G_next = _tail_value(policy, paths, n, params, payoff_fn, disc)
            F = torch.sigmoid(net(S_n))
            reward = disc[n] * g_n * F + G_next * (1.0 - F)
            loss = -reward.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        nets[n] = net                           # policy sees it via shared dict
        prev_state = net.state_dict()
        final_obj[n] = -loss.item()
    train_runtime = time.perf_counter() - t0

    diag = {"steps_per_date": steps_per_date, "batch_size": params.batch_size,
            "warm_start": True, "train_runtime": train_runtime,
            "final_objective_by_date": {k: round(v, 4)
                                        for k, v in sorted(final_obj.items())}}
    return policy, diag


def price(S0, params: Params, payoff_fn=None, device="cpu") -> Result:
    """Train, then value the executed hard policy on fresh independent paths."""
    t0 = time.perf_counter()
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    seed = params.seed if params.seed is not None else 0

    gen_fit = torch.Generator(device=device).manual_seed(seed)
    policy, diag = fit(S0, params, payoff_fn, device, generator=gen_fit)

    from ..duality.andersen_broadie import _policy_payoffs
    gen_eval = torch.Generator(device=device).manual_seed(seed + 2)
    pay = _policy_payoffs(S0, params, policy, payoff_fn, params.n_paths,
                          device, gen_eval)
    value = pay.mean().item()
    stderr = (pay.std() / (params.n_paths ** 0.5)).item()

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return Result(
        value=value,
        runtime=time.perf_counter() - t0,
        diagnostics={"method": "becker (2019 recursive DOS, executed policy)",
                     "mc_stderr": stderr, **diag},
    )
