"""Dai, Sun, Xu and Zhou (2024): penalised Bernoulli stopping, Algorithm 1, with
Ye, Wong and Park's ambiguity term as an option.

- The stopping decision is a Bernoulli control pi in [0, 1]; the penalty K
  makes it a killing rate K pi, with survival weight
      R_{l+1} = R_l (1 - K pi_l dt),  R_0 = 1.                      (3.5)
- The policy is the sigmoid (eq. 2.24)
      pi_l(S) = 1 / (1 + exp( K (V_l(S) - g(S)) / lambda )).
- Training minimises the martingale loss (3.2)-(3.3) over whole paths. With
  D_j = e^{-r t_j},
      G_l = D_T R_T g(X_T)
            + sum_{j >= l} D_j R_j [ K g(X_j) pi_j - lambda H(pi_j)
                                     - eps |Z_j| ] dt
            - D_l R_l V_l(X_l),
      ML  = 1/2 mean_paths sum_l G_l^2 dt,
  with H the Bernoulli negentropy, and one joint Adam step on all networks
  per iteration. Discounting is explicit through D_j, where the paper puts a
  -rV term in the driver.
- eps |Z| is Ye's ambiguity driver, with Z_j = sigma X_j dV_j/dx by
  automatic differentiation. eps = 0 (the default) is Dai's algorithm.
- The temperature anneals geometrically from max(K/10, lambda) to lambda over
  the first 70% of iterations, and the learning rate decays on a cosine
  schedule.

Departure from the paper: Dai learns the early-exercise premium over a
learned European value; here the shared ValueNet is used, as for every
method.

fit() trains and returns the networks; evaluate() values the randomised and
greedy policies on fresh paths; price() runs both.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params
from ..simulate import gbm_paths
from .. import payoffs
from .common import ValueNet

# Penalty K: large enough that raising it leaves the price unchanged
# (the paper uses K = 10).
_DEFAULT_PENALTY = 40.0


def fit(S0, params: Params, payoff_fn=payoffs.put, device=None,
        epsilon: float = 0.0):
    """Train the per-date value networks. Returns (nets, diag)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if params.seed is not None:
        torch.manual_seed(params.seed)

    L, lam_target = params.n_steps, params.lambda_
    K_pen = params.penalty if params.penalty is not None else _DEFAULT_PENALTY
    dt = params.T / L
    D = torch.tensor([math.exp(-params.r * l * dt) for l in range(L + 1)],
                     device=device)                     # discount factors D_l

    nets = [ValueNet(params.dim, params.arch, params.K_strike, payoff_fn).to(device)
            for _ in range(L)]
    all_params = [p for n in nets for p in n.parameters()]
    opt = torch.optim.Adam(all_params, lr=params.lr)    # ONE joint step (Alg. 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=params.n_iters, eta_min=params.lr / 20)

    lam0 = max(K_pen / 10.0, lam_target)
    anneal_iters = max(1, int(0.7 * params.n_iters))

    def train_lambda(it):
        frac = min(1.0, it / anneal_iters)
        return lam0 * (lam_target / lam0) ** frac

    def policy(net, S, g, lam_now):
        """Bernoulli parameter, sigmoid((g - V) * K/lam). Detached."""
        with torch.no_grad():
            return torch.sigmoid(K_pen * (g - net(S)) / lam_now)

    def ambiguity_drift(net, S):
        """eps * |Z| with Z = sigma * S * dV/dS, via autodiff. Detached."""
        S_req = S.detach().requires_grad_(True)
        V = net(S_req)
        grad = torch.autograd.grad(V.sum(), S_req)[0]   # (batch, d)
        Z = params.sigma * S_req * grad
        return (epsilon * Z.norm(dim=-1)).detach()

    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    final_td = None
    for it in range(params.n_iters):
        lam_now = train_lambda(it)
        S_paths = gbm_paths(S0, params, n_paths=params.batch_size, device=device)
        M = S_paths.shape[0]

        R = torch.ones(M, device=device)
        surv, rewards, values = [], [], []
        for l in range(L):
            S_l = S_paths[:, l, :]
            g_l = payoff_fn(S_l, params.K_strike)
            pi = policy(nets[l], S_l, g_l, lam_now) * (g_l > 0)
            H = pi * torch.log(pi.clamp(min=1e-30)) \
                + (1 - pi) * torch.log((1 - pi).clamp(min=1e-30))
            drift = ambiguity_drift(nets[l], S_l) if epsilon > 0 else 0.0
            q = (K_pen * pi * dt).clamp(max=1.0)
            rewards.append(D[l] * R * (g_l * q - (lam_now * H + drift) * dt))
            values.append(D[l] * R * nets[l](S_l))      # grad flows through V only
            surv.append(R)
            R = R * (1.0 - q)

        g_T = payoff_fn(S_paths[:, L, :], params.K_strike)
        terminal = D[L] * R * g_T                       # D_T R_T g(X_T), detached

        rew = torch.stack(rewards)                      # (L, M)
        suffix = torch.flip(torch.cumsum(torch.flip(rew, [0]), 0), [0])
        G = terminal.unsqueeze(0) + suffix - torch.stack(values)   # (L, M)
        loss = 0.5 * (G ** 2).sum(0).mean() * dt        # martingale loss (3.2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        final_td = loss.item()

    if device == "cuda":
        torch.cuda.synchronize()
    diag = {"final_td_error": final_td, "device": device, "penalty_K": K_pen,
            "epsilon": epsilon,
            "train_runtime": time.perf_counter() - t_start,
            "anneal": f"{lam0:g} -> {lam_target:g} over {anneal_iters} iters"}
    return nets, diag


def evaluate(nets, S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Value the randomised and greedy policies on fresh paths, and V_0(S0).
    Does not reseed; call after fit()."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    L, lam_target = params.n_steps, params.lambda_
    K_pen = params.penalty if params.penalty is not None else _DEFAULT_PENALTY
    dt = params.T / L
    D = torch.tensor([math.exp(-params.r * l * dt) for l in range(L + 1)],
                     device=device)

    with torch.no_grad():
        S_test = gbm_paths(S0, params, n_paths=params.n_paths, device=device)
        M = S_test.shape[0]
        alive_r = torch.ones(M, dtype=torch.bool, device=device)
        alive_g = torch.ones(M, dtype=torch.bool, device=device)
        pay_r = torch.zeros(M, device=device)
        pay_g = torch.zeros(M, device=device)
        max_stop_prob = 0.0                             # penalty-saturation check

        otm_stops = 0                                   # diagnostic: stops at g == 0
        for l in range(L):
            S_l = S_test[:, l, :]
            g_l = payoff_fn(S_l, params.K_strike)
            V_l = nets[l](S_l)
            pi = torch.sigmoid(K_pen * (g_l - V_l) / lam_target) * (g_l > 0)
            p_stop = (K_pen * pi * dt).clamp(max=1.0)
            max_stop_prob = max(max_stop_prob, p_stop.max().item())

            stop_r = alive_r & (torch.rand(M, device=device) < p_stop)
            otm_stops += (stop_r & (g_l <= 0)).sum().item()
            pay_r[stop_r] = D[l] * g_l[stop_r]
            alive_r &= ~stop_r

            stop_g = alive_g & (V_l <= g_l) & (g_l > 0)
            pay_g[stop_g] = D[l] * g_l[stop_g]
            alive_g &= ~stop_g

        g_T = payoff_fn(S_test[:, L, :], params.K_strike)
        pay_r[alive_r] = D[L] * g_T[alive_r]
        pay_g[alive_g] = D[L] * g_T[alive_g]

        value = pay_r.mean().item()
        stderr = (pay_r.std() / math.sqrt(M)).item()
        greedy_value = pay_g.mean().item()
        S0_t = torch.full((1, params.dim), float(S0), device=device) \
            if not torch.is_tensor(S0) else torch.as_tensor(S0, device=device).reshape(1, -1)
        V0 = nets[0](S0_t).item()

    return {"value": value, "mc_stderr": stderr,
            "greedy_policy_value": greedy_value, "value_net_V0": V0,
            "max_stop_prob": max_stop_prob, "otm_stop_frac": otm_stops / M}


def price(S0, params: Params, payoff_fn=payoffs.put, device=None,
          epsilon: float = 0.0) -> Result:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    nets, tdiag = fit(S0, params, payoff_fn, device, epsilon=epsilon)
    ev = evaluate(nets, S0, params, payoff_fn, device)

    if device == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t_start

    lam_target = params.lambda_
    return Result(
        value=ev["value"],
        runtime=runtime,
        diagnostics={
            "method": "dai" if epsilon == 0 else f"dai/ye(eps={epsilon:g})",
            "device": device,
            "penalty_K": tdiag["penalty_K"],
            "epsilon": epsilon,
            "final_td_error": tdiag["final_td_error"],
            "mc_stderr": ev["mc_stderr"],
            "greedy_policy_value": ev["greedy_policy_value"],
            "value_net_V0": ev["value_net_V0"],
            "max_stop_prob": ev["max_stop_prob"],
            "otm_stop_frac": ev["otm_stop_frac"],
            "lambda_bias_scale": lam_target * params.T * math.log(2.0),
            "anneal": tdiag["anneal"],
        },
    )
