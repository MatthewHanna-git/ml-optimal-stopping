"""NN-LSM, Lapeyre and Lelong (2021): the Longstaff-Schwartz recursion with a
neural regressor.

The algorithm is that of lsm.py: backward policy iteration on realised cash
flows, in-the-money paths only, one training path set reused across dates.
Only the polynomial basis is replaced, by the shared ValueNet. Against LSM it
changes the function class alone; against RDBDP it changes the target alone
(cash flow against value).

Each date's network starts from the next date's weights, as in the paper.
Departures from the paper: the shared ValueNet trunk in place of their
leaky-ReLU networks, and minibatch steps set by Params in place of epochs.

The exercise rule is: in the money and g >= net(S). price() values the
executed policy on fresh paths; the paper's in-sample estimator, which
reuses the training paths, is kept in diagnostics.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params
from .common import ValueNet


class NNLSMPolicy:
    """Exercise iff in the money and g >= fitted continuation (ValueNet)."""

    def __init__(self, params: Params, payoff_fn, nets):
        self.K = params.K_strike
        self.n_steps = params.n_steps
        self.payoff_fn = payoff_fn
        self.nets = nets                        # {k: ValueNet}, k = 1..N-1
        self.skipped_dates = [k for k in range(1, params.n_steps)
                              if k not in nets]

    @torch.no_grad()
    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        if k not in self.nets:
            return torch.zeros_like(g, dtype=torch.bool)
        return (g > 0.0) & (g >= self.nets[k](S))


def fit(S0, params: Params, payoff_fn=None, device="cpu", n_train=None,
        generator=None):
    """LS backward recursion with ValueNet regression on one fixed path set.
    Returns (NNLSMPolicy, in_sample_value, in_sample_stderr)."""
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    from ..simulate import gbm_paths

    N = params.n_steps
    dt = params.T / N
    n_train = n_train or params.n_paths
    if generator is None:
        seed = params.seed if params.seed is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed)

    paths = gbm_paths(S0, params, n_paths=n_train, device=device,
                      generator=generator)
    disc_step = math.exp(-params.r * dt)
    steps_per_date = max(1, math.ceil(params.n_iters / max(N - 1, 1)))

    cf = payoff_fn(paths[:, N, :], params.K_strike).clone()   # tau = N
    nets = {}
    prev_state = None
    for k in range(N - 1, 0, -1):
        cf *= disc_step                          # discount to date k
        S_k = paths[:, k, :]
        g_k = payoff_fn(S_k, params.K_strike)
        itm = torch.nonzero(g_k > 0.0, as_tuple=True)[0]
        if itm.numel() < 50:                     # too few ITM paths
            continue
        net = ValueNet(params.dim, params.arch, params.K_strike,
                       payoff_fn).to(device)
        if prev_state is not None:
            net.load_state_dict(prev_state)      # warm start, as in the paper
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=params.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_per_date)
        S_itm, cf_itm_t = S_k[itm], cf[itm]
        m = itm.numel()
        for _ in range(steps_per_date):
            idx = torch.randint(m, (min(params.batch_size, m),),
                                device=device, generator=generator)
            pred = net(S_itm[idx])
            loss = ((cf_itm_t[idx] - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        nets[k] = net
        prev_state = net.state_dict()
        with torch.no_grad():
            C = net(S_itm)
        ex = g_k[itm] >= C
        cf_new = cf[itm]
        cf_new[ex] = g_k[itm][ex]
        cf[itm] = cf_new

    in_sample = cf * disc_step                   # discount date 1 -> 0
    policy = NNLSMPolicy(params, payoff_fn, nets)
    return (policy, in_sample.mean().item(),
            (in_sample.std() / (n_train ** 0.5)).item())


def price(S0, params: Params, payoff_fn=None, device="cpu") -> Result:
    """Fit on a training set, then value the EXECUTED policy on fresh paths."""
    t0 = time.perf_counter()
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    seed = params.seed if params.seed is not None else 0

    gen_fit = torch.Generator(device=device).manual_seed(seed)
    policy, v_in, se_in = fit(S0, params, payoff_fn, device, generator=gen_fit)

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
        diagnostics={"method": "nn_lsm (out-of-sample policy value)",
                     "mc_stderr": stderr,
                     "in_sample_value": v_in, "in_sample_stderr": se_in,
                     "skipped_dates": policy.skipped_dates,
                     "warm_start": True},
    )
