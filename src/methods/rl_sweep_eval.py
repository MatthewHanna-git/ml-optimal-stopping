"""The randomised value of each RL method under the certification's
conventions.

Each method's own evaluate() allows exercise at date 0, and Dong's and
Dianetti's randomised rules stop at zero payoff, so their values are not
comparable with the certified L0. This module evaluates the same randomised
rules as certify() executes policies:

- exercise only at dates k = 1, ..., N - 1, and at maturity if in the money;
- never stop where g = 0, with the survival (Dianetti: the fuel) carried
  forward;
- one torch.Generator, seeded by the caller, drives the paths and the
  stopping draws.

Stop probabilities come from each method's intensity at the target
temperature; Dianetti's value uses its exact fuel weights, with no
sampling. The value of the certified deterministic rule on the same paths
is returned as greedy_value.
"""

import math
import time

import torch

from ..simulate import gbm_paths
from . import dong as _dong, dai as _dai, chee as _chee


def _stop_prob(method, nets, k, S, g, params, payoff_fn):
    """Per-date stop probability of the randomised rule at the target dials."""
    lam = params.lambda_
    dt = params.T / params.n_steps
    V = nets[k](S)
    if method == "dong":
        p = _dong._intensity(nets[k], S, lam, payoff_fn, params.K_strike,
                             math.log(1.0 / dt)) * dt
        return p.clamp(max=1.0)
    if method == "dai":
        K_pen = params.penalty if params.penalty is not None else _dai._DEFAULT_PENALTY
        pi = torch.sigmoid(K_pen * (g - V) / lam)
        return (K_pen * pi * dt).clamp(max=1.0)
    if method == "chee":
        n = 1.0 / lam
        mu = _chee._mu((g - V) / lam, n)
        return 1.0 - torch.exp(-mu * dt)
    raise ValueError(method)


@torch.no_grad()
def randomised_value(method, nets, S0, params, payoff_fn, device, seed,
                     n_paths=None, threshold=math.exp(-1.0)):
    """Randomised-rule value under certification's conventions.

    Returns dict(value, mc_stderr, greedy_value, otm_stop_frac, n_paths,
    eval_s). eval_s is CUDA-synchronised wall time of this call."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    N = params.n_steps
    dt = params.T / N
    n_paths = n_paths or params.n_paths
    D = [math.exp(-params.r * k * dt) for k in range(N + 1)]
    gen = torch.Generator(device=device).manual_seed(int(seed))

    S = gbm_paths(S0, params, n_paths=n_paths, device=device, generator=gen)
    M = S.shape[0]
    pay_r = torch.zeros(M, device=device)
    pay_g = torch.zeros(M, device=device)
    alive_g = torch.ones(M, dtype=torch.bool, device=device)
    otm_stops = 0.0

    if method == "dianetti":
        Y = torch.ones(M, device=device)                   # exact survival
        for k in range(1, N):
            S_k = S[:, k, :]
            g_k = payoff_fn(S_k, params.K_strike)
            itm = g_k > 0.0
            b = nets[k](S_k)
            spend = torch.relu(Y - b) * itm                # guard: no spend at g = 0
            pay_r += D[k] * g_k * spend
            Y = Y - spend
            stop_g = alive_g & (b <= threshold) & itm
            pay_g[stop_g] = D[k] * g_k[stop_g]
            alive_g &= ~stop_g
        g_T = payoff_fn(S[:, N, :], params.K_strike)
        pay_r += D[N] * g_T * Y                            # forced at maturity
        pay_g[alive_g] = D[N] * g_T[alive_g]
    else:
        alive_r = torch.ones(M, dtype=torch.bool, device=device)
        for k in range(1, N):
            S_k = S[:, k, :]
            g_k = payoff_fn(S_k, params.K_strike)
            itm = g_k > 0.0
            p = _stop_prob(method, nets, k, S_k, g_k, params, payoff_fn) * itm
            u = torch.rand(M, device=device, generator=gen)
            stop_r = alive_r & (u < p)
            otm_stops += (stop_r & ~itm).sum().item()      # 0 under the guard
            pay_r[stop_r] = D[k] * g_k[stop_r]
            alive_r &= ~stop_r
            stop_g = alive_g & (nets[k](S_k) <= g_k) & itm
            pay_g[stop_g] = D[k] * g_k[stop_g]
            alive_g &= ~stop_g
        g_T = payoff_fn(S[:, N, :], params.K_strike)
        pay_r[alive_r] = D[N] * g_T[alive_r]
        pay_g[alive_g] = D[N] * g_T[alive_g]

    if device == "cuda":
        torch.cuda.synchronize()
    return {"value": pay_r.mean().item(),
            "mc_stderr": (pay_r.std() / math.sqrt(M)).item(),
            "greedy_value": pay_g.mean().item(),
            "otm_stop_frac": otm_stops / M,
            "n_paths": M,
            "eval_s": time.perf_counter() - t0}
