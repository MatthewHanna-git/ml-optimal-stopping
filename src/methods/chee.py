"""Chee, Frikha and Li (2026): entropy-regularised penalisation by density
control, the policy improvement algorithm (arXiv 2602.18078).

- The control is a probability density over intensities u in [0, n]. The
  optimal density is Gibbs, pi*(u) proportional to exp((P - V) u / lambda)
  (eq. 3.4).
- The driver is linear in u, so the density acts only through its mean
  (eq. 5.3),
      mu(alpha, n) = n / (1 - e^{-alpha n}) - 1/alpha,  alpha = (P - V)/lambda,
  and the executed policy is a Poisson stop with intensity mu. The
  regulariser leaves the special function Phi(x) = ln((e^x - 1)/x)
  (eq. 3.5), evaluated at x = n alpha.
- The truncation is tied to the temperature, n = 1/lambda, as in the paper;
  params.penalty is ignored.

The scheme (eqs. 5.2-5.3), one network per date:
    V_k = e^{-a dt} E[ V_{k+1} | F_k ] + (b/a) (1 - e^{-a dt}),
    a   = mu(alpha, n) + r,
    b   = lambda Phi(n alpha) + lambda ln(n) + V (a - r),
with P the undiscounted payoff and discounting entering through r in a.

Departure from the paper: the conditional expectations are fitted by the
shared ValueNet in place of least-squares regression on the 13
Andersen-Broadie basis functions. As in dong.py, each iteration takes one
Adam step per date against targets built from the current networks.

The temperature anneals geometrically from max(0.1, lambda) to lambda over
the first 70% of iterations, unless Params.lambda_start is set, with n =
1/lambda throughout; the learning rate decays on a cosine schedule. The
executed policy never stops at zero payoff; the training targets are
unmasked.

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


# ---------------------------------------------------- special functions ---

def _phi(x):
    """Phi(x) = ln((e^x - 1)/x), numerically stable on the whole line."""
    small = x.abs() < 1e-4
    xs = torch.where(small, torch.ones_like(x), x)      # safe placeholder
    pos = xs > 0
    xp = xs.clamp(min=1e-4)                              # for the x>0 branch
    xn = xs.clamp(max=-1e-4)                             # for the x<0 branch
    phi_pos = xp + torch.log1p(-torch.exp(-xp)) - torch.log(xp)
    phi_neg = torch.log1p(-torch.exp(xn)) - torch.log(-xn)
    out = torch.where(pos, phi_pos, phi_neg)
    return torch.where(small, x / 2 + x ** 2 / 24, out)  # series near 0


def _mu(alpha, n):
    """mu(alpha, n) = n/(1 - e^{-alpha n}) - 1/alpha, stable on the whole line.

    Limits: n/2 at alpha = 0; n as alpha -> +inf; -1/alpha -> 0+ as
    alpha -> -inf, a power-law tail where Dong's is exponential."""
    z = alpha * n
    small = z.abs() < 1e-4
    zs = torch.where(small, torch.ones_like(z), z).clamp(-30.0, 30.0)
    h_mid = 1.0 / (1.0 - torch.exp(-zs)) - 1.0 / zs
    h = torch.where(z > 30.0, 1.0 - 1.0 / z,
        torch.where(z < -30.0, -1.0 / z, h_mid))
    h = torch.where(small, 0.5 + z / 12.0, h)
    return n * h


def fit(S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Train the per-date networks. Returns (nets, diag)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if params.seed is not None:
        torch.manual_seed(params.seed)

    L, lam_target = params.n_steps, params.lambda_
    dt = params.T / L

    nets = [ValueNet(params.dim, params.arch, params.K_strike, payoff_fn).to(device)
            for _ in range(L)]
    opts = [torch.optim.Adam(m.parameters(), lr=params.lr) for m in nets]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(
        o, T_max=params.n_iters, eta_min=params.lr / 20) for o in opts]

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
        n_now = 1.0 / lam_now                  # n = 1/lambda at every temperature
        S_paths = gbm_paths(S0, params, n_paths=params.batch_size, device=device)
        td_sum = 0.0
        for k in range(L - 1, -1, -1):
            S_k, S_next = S_paths[:, k, :], S_paths[:, k + 1, :]
            P_k = payoff_fn(S_k, params.K_strike)

            with torch.no_grad():              # policy + targets, all detached
                V_k = nets[k](S_k)
                V_next = payoff_fn(S_next, params.K_strike) if k == L - 1 \
                    else nets[k + 1](S_next)
                alpha = (P_k - V_k) / lam_now
                a = _mu(alpha, n_now) + params.r                    # (5.2)
                b = (lam_now * _phi(n_now * alpha)
                     + lam_now * math.log(n_now)
                     + V_k * (a - params.r))
                decay = torch.exp(-a * dt)
                target = decay * V_next + (b / a) * (1.0 - decay)

            loss = ((target - nets[k](S_k)) ** 2).mean()
            opts[k].zero_grad()
            loss.backward()
            opts[k].step()
            scheds[k].step()
            td_sum += loss.item()
        final_td = td_sum / L

    if device == "cuda":
        torch.cuda.synchronize()
    diag = {"final_td_error": final_td, "device": device,
            "n_truncation": 1.0 / lam_target,
            "train_runtime": time.perf_counter() - t_start,
            "anneal": f"{lam0:g} -> {lam_target:g} over {anneal_iters} iters"}
    return nets, diag


def evaluate(nets, S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Value the trained policy on fresh paths. Does not reseed; call after
    fit()."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    L, lam_target = params.n_steps, params.lambda_
    n_target = 1.0 / lam_target
    dt = params.T / L

    with torch.no_grad():
        S_test = gbm_paths(S0, params, n_paths=params.n_paths, device=device)
        M = S_test.shape[0]
        alive_r = torch.ones(M, dtype=torch.bool, device=device)
        alive_g = torch.ones(M, dtype=torch.bool, device=device)
        pay_r = torch.zeros(M, device=device)
        pay_g = torch.zeros(M, device=device)
        otm_stops = 0

        for k in range(L):
            S_k = S_test[:, k, :]
            P_k = payoff_fn(S_k, params.K_strike)
            V_k = nets[k](S_k)
            df = math.exp(-params.r * k * dt)

            alpha = (P_k - V_k) / lam_target
            mu = _mu(alpha, n_target) * (P_k > 0)          # never stop at zero payoff
            p_stop = 1.0 - torch.exp(-mu * dt)

            stop_r = alive_r & (torch.rand(M, device=device) < p_stop)
            otm_stops += (stop_r & (P_k <= 0)).sum().item()
            pay_r[stop_r] = df * P_k[stop_r]
            alive_r &= ~stop_r

            stop_g = alive_g & (V_k <= P_k) & (P_k > 0)
            pay_g[stop_g] = df * P_k[stop_g]
            alive_g &= ~stop_g

        P_T = payoff_fn(S_test[:, L, :], params.K_strike)
        df_T = math.exp(-params.r * params.T)
        pay_r[alive_r] = df_T * P_T[alive_r]
        pay_g[alive_g] = df_T * P_T[alive_g]

        value = pay_r.mean().item()
        stderr = (pay_r.std() / math.sqrt(M)).item()
        greedy_value = pay_g.mean().item()
        S0_t = torch.full((1, params.dim), float(S0), device=device) \
            if not torch.is_tensor(S0) else torch.as_tensor(S0, device=device).reshape(1, -1)
        V0 = nets[0](S0_t).item()

    return {"value": value, "mc_stderr": stderr,
            "greedy_policy_value": greedy_value, "value_net_V0": V0,
            "otm_stop_frac": otm_stops / M}


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
            "method": "chee",
            "device": device,
            "n_truncation": tdiag["n_truncation"],
            "final_td_error": tdiag["final_td_error"],
            "mc_stderr": ev["mc_stderr"],
            "greedy_policy_value": ev["greedy_policy_value"],
            "value_net_V0": ev["value_net_V0"],
            "otm_stop_frac": ev["otm_stop_frac"],
            "lambda_bias_scale": -lam_target * math.log(lam_target),
            "anneal": tdiag["anneal"],
        },
    )
