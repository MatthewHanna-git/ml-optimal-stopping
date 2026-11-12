"""Dong (2024): entropy-regularised randomised stopping by policy iteration,
Algorithm 4.1.

- The policy is a stopping intensity in Gibbs form (eq. 3.6),
      pi_l(S) = exp( -(V_l(S) - g(S)) / lambda ),
  with regulariser R(pi) = pi - pi log pi.
- One value network per date, trained backwards in time on the TD error
  (eq. 4.4),
      delta_l = ( g pi dt + lambda R(pi) dt
                  + e^{-r dt} V_{l+1}(S_{l+1}) (1 - pi dt) - V_l(S_l) )^2,
  with one gradient step per date per iteration on a fresh batch of paths.
- pi and V_{l+1} are held fixed in the TD target, so each step evaluates the
  current policy, and recomputing pi from the updated V improves it.
- pi is capped so that pi dt <= 1, keeping (1 - pi dt) a survival
  probability; the cap is applied inside the exponent.
- The temperature anneals geometrically from max(0.1, lambda) to lambda over
  the first 70% of iterations, unless Params.lambda_start is set, and the
  learning rate decays on a cosine schedule.

fit() trains and returns the networks; evaluate() values on fresh paths the
randomised policy (the paper's reported price), the greedy policy (stop the
first time V <= g, eq. 3.8) and the network's own V_0(S0). price() runs both.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params
from ..simulate import gbm_paths
from .. import payoffs
from .common import ValueNet


def _intensity(net, S, lam_now, payoff_fn, K_strike, max_exponent):
    """Gibbs policy pi = exp(-(V-g)/lam), clamped so pi*dt <= 1. Detached."""
    with torch.no_grad():
        expo = -(net(S) - payoff_fn(S, K_strike)) / lam_now
        return torch.exp(torch.clamp(expo, max=max_exponent))


def fit(S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Train the per-date value networks. Returns (nets, diag)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if params.seed is not None:
        torch.manual_seed(params.seed)

    L, lam = params.n_steps, params.lambda_
    dt = params.T / L
    disc = math.exp(-params.r * dt)
    max_exponent = math.log(1.0 / dt)

    nets = [ValueNet(params.dim, params.arch, params.K_strike, payoff_fn).to(device)
            for _ in range(L)]                   # l = 0 .. L-1; V_L == g exactly
    opts = [torch.optim.Adam(n.parameters(), lr=params.lr) for n in nets]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(
        o, T_max=params.n_iters, eta_min=params.lr / 20) for o in opts]

    lam_target = lam
    # Annealing start: max(0.1, lambda) unless Params.lambda_start is set.
    lam0 = (max(0.1, lam_target) if getattr(params, 'lambda_start', None) is None
            else float(params.lambda_start))
    anneal_iters = max(1, int(0.7 * params.n_iters))

    def train_lambda(it):
        frac = min(1.0, it / anneal_iters)
        return lam0 * (lam_target / lam0) ** frac

    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    final_td = None
    for it in range(params.n_iters):
        lam_now = train_lambda(it)
        S_paths = gbm_paths(S0, params, n_paths=params.batch_size, device=device)
        td_sum = 0.0
        for l in range(L - 1, -1, -1):           # backward in time, as in Alg. 4.1
            S_l, S_next = S_paths[:, l, :], S_paths[:, l + 1, :]
            g_l = payoff_fn(S_l, params.K_strike)

            pi = _intensity(nets[l], S_l, lam_now, payoff_fn, params.K_strike,
                            max_exponent)                         # policy update (3.6)
            with torch.no_grad():                                 # V_{l+1} target, detached
                V_next = payoff_fn(S_next, params.K_strike) if l == L - 1 \
                    else nets[l + 1](S_next)
                R = pi - pi * torch.log(pi.clamp(min=1e-30))      # R(pi) = pi - pi log pi
                target = (g_l * pi * dt + lam_now * R * dt
                          + disc * V_next * (1.0 - pi * dt))

            loss = ((target - nets[l](S_l)) ** 2).mean()          # TD error (4.4)
            opts[l].zero_grad()
            loss.backward()
            opts[l].step()
            scheds[l].step()
            td_sum += loss.item()
        final_td = td_sum / L

    if device == "cuda":
        torch.cuda.synchronize()
    diag = {"final_td_error": final_td, "device": device,
            "train_runtime": time.perf_counter() - t_start,
            "anneal": f"{lam0:g} -> {lam_target:g} over {anneal_iters} iters"}
    return nets, diag


def evaluate(nets, S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Value the randomised and greedy policies on fresh paths, and V_0(S0).
    Does not reseed; call after fit()."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    L, lam_target = params.n_steps, params.lambda_
    dt = params.T / L
    max_exponent = math.log(1.0 / dt)

    with torch.no_grad():
        S_test = gbm_paths(S0, params, n_paths=params.n_paths, device=device)
        M = S_test.shape[0]
        alive_rand = torch.ones(M, dtype=torch.bool, device=device)
        alive_greedy = torch.ones(M, dtype=torch.bool, device=device)
        pay_rand = torch.zeros(M, device=device)
        pay_greedy = torch.zeros(M, device=device)

        for l in range(L):
            S_l = S_test[:, l, :]
            g_l = payoff_fn(S_l, params.K_strike)
            V_l = nets[l](S_l)
            df = math.exp(-params.r * l * dt)

            p_stop = (_intensity(nets[l], S_l, lam_target, payoff_fn,
                                 params.K_strike, max_exponent) * dt).clamp(max=1.0)
            stop_r = alive_rand & (torch.rand(M, device=device) < p_stop)
            pay_rand[stop_r] = df * g_l[stop_r]
            alive_rand &= ~stop_r

            stop_g = alive_greedy & (V_l <= g_l) & (g_l > 0)      # tau^lambda of (3.8)
            pay_greedy[stop_g] = df * g_l[stop_g]
            alive_greedy &= ~stop_g

        g_T = payoff_fn(S_test[:, L, :], params.K_strike)
        df_T = math.exp(-params.r * params.T)
        pay_rand[alive_rand] = df_T * g_T[alive_rand]
        pay_greedy[alive_greedy] = df_T * g_T[alive_greedy]

        value = pay_rand.mean().item()
        stderr = (pay_rand.std() / math.sqrt(M)).item()
        greedy_value = pay_greedy.mean().item()
        S0_t = torch.full((1, params.dim), float(S0), device=device) \
            if not torch.is_tensor(S0) else torch.as_tensor(S0, device=device).reshape(1, -1)
        V0 = nets[0](S0_t).item()

    return {"value": value, "mc_stderr": stderr,
            "greedy_policy_value": greedy_value, "value_net_V0": V0}


def price(S0, params: Params, payoff_fn=payoffs.put, device=None) -> Result:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    nets, tdiag = fit(S0, params, payoff_fn, device)
    ev = evaluate(nets, S0, params, payoff_fn, device)

    if device == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t_start

    lam_target = params.lambda_
    return Result(
        value=ev["value"],
        runtime=runtime,
        diagnostics={
            "method": "dong",
            "device": device,
            "final_td_error": tdiag["final_td_error"],
            "mc_stderr": ev["mc_stderr"],
            "greedy_policy_value": ev["greedy_policy_value"],
            "value_net_V0": ev["value_net_V0"],
            "lambda_bias_scale": -lam_target * math.log(lam_target),
            "anneal": tdiag["anneal"],
        },
    )
