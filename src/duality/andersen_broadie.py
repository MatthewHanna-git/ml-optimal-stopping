"""The Andersen-Broadie (2004) primal-dual interval for an exercise policy.

Given any policy (see policies.Policy), certify() returns
    lower bound  L0 = the value of executing the policy, by Monte Carlo,
    upper bound  U0 = L0 + Delta0, with Delta0 = E[max_k (h_k/B_k - pi_k)],
and a conservative 95% confidence interval for the Bermudan price
(AB eq. 16). The martingale pi is built from the policy's own value
process (AB eq. 9) by nested simulation: N1 outer paths and, at each date,
N2 inner subpaths that execute the same policy. Inner-simulation noise
biases Delta0 upwards, so the interval is valid at any inner budget.

Conventions:
- Exercise dates are k = 1, ..., n_steps at t_k = k T / n_steps; date 0 is
  excluded.
- Values are in time-0 dollars: h_k/B_k = e^{-r t_k} g(S_k).
- No policy exercises at zero payoff: the g > 0 guard is applied here, to
  every policy.
- The lower bound and the dual increment use independent path sets, seeded
  from seed + 7919 and seed + 7920.

Returns a Result with value = L0 + Delta0/2 (AB eq. 17), bounds = (L0, U0),
and the standard errors, the 95% interval and the per-date exercise
fractions in diagnostics.
"""

import math
import time

import torch

from ..params import Params
from ..result import Result


def _guarded_exercise(policy, k, S, payoff_fn, K, n_steps):
    """Policy decision with the central g > 0 guard; maturity = collect payoff."""
    g = payoff_fn(S, K)
    if k == n_steps:
        return g > 0.0, g
    return policy.exercise(k, S) & (g > 0.0), g


def _policy_payoffs(S0, params, policy, payoff_fn, n_paths, device, generator):
    """Discounted payoff of executing the policy on n_paths fresh paths: (n_paths,)."""
    from ..simulate import gbm_paths
    N = params.n_steps
    dt = params.T / N

    # If the full path tensor would exceed about 1 GB (from d = 20 at 2^21
    # paths), generate and execute the paths in chunks.
    bytes_full = 4 * n_paths * (N + 1) * params.dim
    if bytes_full <= 2 ** 30:
        chunks = [n_paths]
    else:
        per = max(1, min(n_paths, 2 ** 30 // (4 * (N + 1) * params.dim)))
        chunks = [per] * (n_paths // per)
        if n_paths % per:
            chunks.append(n_paths % per)

    out = []
    for m in chunks:
        paths = gbm_paths(S0, params, n_paths=m, device=device,
                          generator=generator)
        payoff = torch.zeros(m, device=device)
        alive = torch.ones(m, dtype=torch.bool, device=device)
        for k in range(1, N + 1):
            ex, g = _guarded_exercise(policy, k, paths[:, k, :], payoff_fn,
                                      params.K_strike, N)
            stop = alive & ex
            payoff[stop] = math.exp(-params.r * k * dt) * g[stop]
            alive &= ~ex
        out.append(payoff)
    return out[0] if len(out) == 1 else torch.cat(out)


def _inner_continuation(S_start, k_from, params, policy, payoff_fn, n_inner,
                        device, generator):
    """C_k: time-0-discounted value of continuing at date k_from and following
    the policy from k_from + 1 on. S_start: (m, dim). Returns (m,) means and
    the per-start standard error implied by n_inner subpaths."""
    N, d = params.n_steps, params.dim
    dt = params.T / N
    drift = (params.r - params.delta - 0.5 * params.sigma ** 2) * dt
    vol = params.sigma * (dt ** 0.5)

    m = S_start.shape[0]
    S = S_start.repeat_interleave(n_inner, dim=0)            # (m * n_inner, d)
    M = S.shape[0]
    payoff = torch.zeros(M, device=device)
    alive = torch.ones(M, dtype=torch.bool, device=device)
    for k in range(k_from + 1, N + 1):
        Z = torch.randn(M, d, device=device, generator=generator)
        S = S * torch.exp(drift + vol * Z)
        ex, g = _guarded_exercise(policy, k, S, payoff_fn, params.K_strike, N)
        stop = alive & ex
        payoff[stop] = math.exp(-params.r * k * dt) * g[stop]
        alive &= ~ex
    payoff = payoff.view(m, n_inner)
    return payoff.mean(dim=1), payoff.std(dim=1) / (n_inner ** 0.5)


def certify(S0, params: Params, policy, payoff_fn=None, device="cpu",
            chunk_outer=None) -> Result:
    """Run the full Andersen-Broadie certification of `policy`.

    chunk_outer: outer paths processed per inner-simulation batch (memory
    control; chunk_outer * n_inner states are alive at once). Default sizes
    the batch to ~2^22 states.
    """
    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.put if params.dim == 1 else payoffs.max_call
    t_start = time.perf_counter()

    N = params.n_steps
    dt = params.T / N
    N1, N2 = params.n_outer, params.n_inner
    if chunk_outer is None:
        chunk_outer = max(1, (2 ** 22) // max(N2, 1))

    # The offsets keep the certification paths disjoint from the training
    # paths, which fit() draws from seed onwards; otherwise the lower bound
    # would be evaluated in sample.
    seed = params.seed if params.seed is not None else 0
    gen_lower = torch.Generator(device=device).manual_seed(seed + 7919)
    gen_dual = torch.Generator(device=device).manual_seed(seed + 7920)

    # ---- lower bound: independent path set ---------------------------------
    pay = _policy_payoffs(S0, params, policy, payoff_fn, params.n_paths,
                          device, gen_lower)
    L0 = pay.mean().item()
    sL = (pay.std() / (params.n_paths ** 0.5)).item()

    # ---- dual increment: N1 outer paths, nested inner simulations ----------
    from ..simulate import gbm_paths
    outer = gbm_paths(S0, params, n_paths=N1, device=device, generator=gen_dual)

    disc = torch.exp(-params.r * dt * torch.arange(N + 1, device=device))
    h_disc = torch.zeros(N1, N + 1, device=device)     # h_k / B_k on outer paths
    L_disc = torch.zeros(N1, N + 1, device=device)     # L_k / B_k on outer paths
    adj = torch.zeros(N1, N + 1, device=device)        # l_k * (C_k - L_k/B_k)
    ex_frac = []

    for k in range(1, N + 1):
        S_k = outer[:, k, :]
        ex, g = _guarded_exercise(policy, k, S_k, payoff_fn, params.K_strike, N)
        h_disc[:, k] = disc[k] * g
        ex_frac.append(ex.float().mean().item())
        if k == N:
            L_disc[:, k] = h_disc[:, k]                # tau_N = N: no inner sim
            continue
        C_k = torch.zeros(N1, device=device)
        for lo in range(0, N1, chunk_outer):
            hi = min(lo + chunk_outer, N1)
            C_k[lo:hi], _ = _inner_continuation(S_k[lo:hi], k, params, policy,
                                                payoff_fn, N2, device, gen_dual)
        L_disc[:, k] = torch.where(ex, h_disc[:, k], C_k)
        adj[:, k] = torch.where(ex, C_k - h_disc[:, k],
                                torch.zeros(N1, device=device))

    # AB eq. 9: pi_1 = L_1/B_1; pi_k = pi_{k-1} + (L_k - L_{k-1})/B - adj_{k-1}
    pi = torch.zeros(N1, N + 1, device=device)
    pi[:, 1] = L_disc[:, 1]
    for k in range(2, N + 1):
        pi[:, k] = pi[:, k - 1] + L_disc[:, k] - L_disc[:, k - 1] - adj[:, k - 1]

    gap_path = (h_disc[:, 1:] - pi[:, 1:]).max(dim=1).values
    delta = gap_path.mean().item()
    s_delta = (gap_path.std() / (N1 ** 0.5)).item()

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t_start

    U0 = L0 + delta
    z = 1.959963984540054
    ci = (L0 - z * sL, U0 + z * (sL ** 2 + s_delta ** 2) ** 0.5)
    g0 = float(payoff_fn(torch.as_tensor([S0] * params.dim, dtype=torch.float32
                                         ).unsqueeze(0), params.K_strike)[0])

    return Result(
        value=L0 + 0.5 * delta,
        runtime=runtime,
        bounds=(L0, U0),
        diagnostics={
            "method": "andersen_broadie",
            "lower": L0, "lower_stderr": sL,
            "delta": delta, "delta_stderr": s_delta,
            "upper": U0,
            "ci95": ci,
            "gap": U0 - L0,
            "n_lower_paths": params.n_paths, "n_outer": N1, "n_inner": N2,
            "exercise_fraction_by_date": ex_frac,
            "t0_payoff": g0,        # exercising at t=0 excluded; compare vs L0
        },
    )
