"""Reflected deep backward dynamic programming (RDBDP), Huré, Pham and Warin
(2020), scheme (3.10).

At each date a pair of networks (U_k, Z_k) approximates the continuation
value and its martingale loading Z = sigma^T D_x u. They are trained
backwards by regressing the next date's reflected value onto
U_k(S_k) + Z_k(S_k).dW:

    loss_k = E | max(U_{k+1}(S_{k+1}), g_{k+1}) - U_k(S_k) - Z_k(S_k).dW_k |^2

with values discounted to time 0, so the driver vanishes. Z explains the
martingale part of the increment, which sharpens the value fit.

Departures from the paper:
- Exact GBM steps in place of the Euler scheme, with the Brownian increment
  recovered from the path:
  dW = (log(S_{k+1}/S_k) - (r - delta - sigma^2/2) dt) / sigma.
- The shared trunks, ValueNet for U and GradNet for Z, in place of the
  paper's (d + 10)-neuron tanh networks.
- Params.n_iters is the total budget, split across the N - 1 trainable
  dates; Adam at Params.lr, cosine-decayed per date. Warm starts across
  dates follow the paper.

The exercise rule is g_k >= U_k. price() values the executed policy on fresh
paths and reports the scheme's own estimate, max(g_0, E[reflected value at
date 1]), as value_net_V0.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params
from .common import ValueNet, GradNet


class HurePolicy:
    """Exercise rule: stop at date k iff the (discounted) payoff is at least
    the U_k continuation estimate. Comparison in time-0 discounted units."""

    def __init__(self, params: Params, payoff_fn, nets, disc):
        self.K = params.K_strike
        self.n_steps = params.n_steps
        self.payoff_fn = payoff_fn
        self.nets = nets                        # {k: ValueNet}, k = 1..N-1
        self.disc = disc

    @torch.no_grad()
    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        return (self.disc[k] * g >= self.nets[k](S)) & (g > 0.0)


def fit(S0, params: Params, payoff_fn=None, device="cpu", generator=None):
    """Backward RDBDP training. Returns (HurePolicy, diagnostics)."""
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    from ..simulate import gbm_paths

    N = params.n_steps
    dt = params.T / N
    disc = [math.exp(-params.r * k * dt) for k in range(N + 1)]
    drift = (params.r - params.delta - 0.5 * params.sigma ** 2) * dt
    if generator is None:
        seed = params.seed if params.seed is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed)

    steps_per_date = max(1, math.ceil(params.n_iters / max(N - 1, 1)))
    u_nets, z_nets = {}, {}
    final_loss = {}

    def reflected_next(k_next, S_next):
        """Time-0-discounted reflected value at date k_next (no grad)."""
        with torch.no_grad():
            g = disc[k_next] * payoff_fn(S_next, params.K_strike)
            if k_next == N:
                return g
            return torch.maximum(u_nets[k_next](S_next), g)

    t0 = time.perf_counter()
    prev_u, prev_z = None, None
    for k in range(N - 1, 0, -1):
        u_net = ValueNet(params.dim, params.arch, params.K_strike,
                         payoff_fn).to(device)
        z_net = GradNet(params.dim, params.arch, params.K_strike,
                        payoff_fn).to(device)
        if prev_u is not None:
            u_net.load_state_dict(prev_u)       # warm start, as in the paper
            z_net.load_state_dict(prev_z)
        opt = torch.optim.Adam(list(u_net.parameters())
                               + list(z_net.parameters()), lr=params.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_per_date)
        for _ in range(steps_per_date):
            paths = gbm_paths(S0, params, n_paths=params.batch_size,
                              device=device, generator=generator)
            S_k, S_k1 = paths[:, k, :], paths[:, k + 1, :]
            dW = (torch.log(S_k1 / S_k) - drift) / params.sigma  # (batch, d)
            target = reflected_next(k + 1, S_k1)
            pred = u_net(S_k) + (z_net(S_k) * dW).sum(dim=-1)
            loss = ((target - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        u_net.eval(), z_net.eval()
        u_nets[k], z_nets[k] = u_net, z_net
        prev_u, prev_z = u_net.state_dict(), z_net.state_dict()
        final_loss[k] = loss.item()
    train_runtime = time.perf_counter() - t0

    policy = HurePolicy(params, payoff_fn, u_nets, disc)
    diag = {"steps_per_date": steps_per_date, "batch_size": params.batch_size,
            "warm_start": True, "train_runtime": train_runtime,
            "final_loss_by_date": {k: round(v, 5)
                                   for k, v in sorted(final_loss.items())}}
    return policy, diag


def price(S0, params: Params, payoff_fn=None, device="cpu") -> Result:
    """Train, then value the executed policy on fresh independent paths."""
    t0 = time.perf_counter()
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    seed = params.seed if params.seed is not None else 0

    gen_fit = torch.Generator(device=device).manual_seed(seed)
    policy, diag = fit(S0, params, payoff_fn, device, generator=gen_fit)

    from ..duality.andersen_broadie import _policy_payoffs
    from ..simulate import gbm_paths
    gen_eval = torch.Generator(device=device).manual_seed(seed + 2)
    pay = _policy_payoffs(S0, params, policy, payoff_fn, params.n_paths,
                          device, gen_eval)
    value = pay.mean().item()
    stderr = (pay.std() / (params.n_paths ** 0.5)).item()

    # scheme-native estimate: max(g(0), E[reflected value at date 1])
    with torch.no_grad():
        gen_v0 = torch.Generator(device=device).manual_seed(seed + 3)
        p1 = gbm_paths(S0, params, n_paths=2 ** 17, device=device,
                       generator=gen_v0)[:, 1, :]
        g1 = math.exp(-params.r * params.T / params.n_steps) \
            * payoff_fn(p1, params.K_strike)
        cont0 = (torch.maximum(policy.nets[1](p1), g1).mean().item()
                 if params.n_steps > 1 else g1.mean().item())
        S0_t = torch.as_tensor([S0] * params.dim,
                               dtype=torch.float32).unsqueeze(0)
        v0 = max(float(payoff_fn(S0_t, params.K_strike)[0]), cont0)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return Result(
        value=value,
        runtime=time.perf_counter() - t0,
        diagnostics={"method": "hure (RDBDP, executed policy)",
                     "mc_stderr": stderr, "value_net_V0": v0, **diag},
    )
