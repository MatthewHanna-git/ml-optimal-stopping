"""Dianetti, Ferrari and Xu (2024): exploratory stopping as finite-fuel
singular control, on the Bermudan grid.

The regularised problem is the paper's; the optimiser is not. Here the
control does not affect the state, so the objective is pathwise
differentiable in the policy and is maximised by direct policy gradient on
simulated paths, in place of the paper's model-free actor-critic.

- The extended state is (X, Y), with Y the remaining stopping mass ("fuel"),
  Y_0 = 1. The policy is a fuel boundary b_k(x) in [0, 1], one BoundaryNet
  per date, and fuel is spent by reflection:
      spend_k = (Y_k - b_k(X_k))^+,  Y_{k+1} = Y_k - spend_k,
  with the remaining fuel spent at maturity.
- The regularised objective (eq. 4.13, with no running profit):
      J = E[ sum_k e^{-r t_k} ( g(X_k) spend_k + lambda (-Y_k log Y_k) dt )
             + e^{-rT} g(X_T) Y_T ].
  The cumulative residual entropy -Y log Y rewards holding Y near 1/e, so
  the regularised optimum may spend fuel at zero payoff. No g > 0 mask is
  applied; otm_spend_frac measures this, and it vanishes as lambda -> 0.
- Y is the distribution function of the randomised stopping time, so each
  test path contributes its exact fuel-weighted payoff, with no sampling.

Training: one joint Adam step over all boundary networks per iteration on a
fresh antithetic batch; the temperature anneals geometrically from
max(0.1, lambda) to lambda over the first 70% of iterations, and the
learning rate decays on a cosine schedule. The reflection is smoothed by a
softplus whose width anneals from 0.1 to 1e-3; evaluation uses hard
reflection. The joint objective is harder to optimise than the per-date
regressions of the other methods, so it is run at a larger budget and
learning rate (6000 iterations of 2048 paths at lr 5e-3).

The reported value is payoff-only, as for every method; the
entropy-included objective J is kept in diagnostics.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params
from ..simulate import gbm_paths
from .. import payoffs
from .common import BoundaryNet


def _rollout(nets, S_paths, params, payoff_fn, device, lam_now, smooth=0.0):
    """Roll the fuel process along given paths. Returns per-path
    (payoff value, entropy value, fuel spent out of the money, Y_T)."""
    L = params.n_steps
    dt = params.T / L
    D = [math.exp(-params.r * l * dt) for l in range(L + 1)]
    M = S_paths.shape[0]
    Y = torch.ones(M, device=device)
    pay = torch.zeros(M, device=device)
    ent = torch.zeros(M, device=device)
    otm = torch.zeros(M, device=device)
    for k in range(L):
        S_k = S_paths[:, k, :]
        g_k = payoff_fn(S_k, params.K_strike)
        b = nets[k](S_k)
        if smooth > 0:
            spend = smooth * torch.nn.functional.softplus((Y - b) / smooth)
            spend = torch.minimum(spend, Y)          # never overspend the fuel
        else:
            spend = torch.relu(Y - b)
        pay = pay + D[k] * g_k * spend
        ent = ent + D[k] * (-Y * torch.log(Y.clamp(min=1e-12))) * dt
        otm = otm + spend * (g_k <= 0)
        Y = Y - spend
    g_T = payoff_fn(S_paths[:, L, :], params.K_strike)
    pay = pay + D[L] * g_T * Y
    return pay, ent, otm, Y


def fit(S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Train the per-date networks. Returns (nets, diag)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if params.seed is not None:
        torch.manual_seed(params.seed)

    L, lam_target = params.n_steps, params.lambda_

    nets = [BoundaryNet(params.dim, params.arch, params.K_strike, payoff_fn).to(device)
            for _ in range(L)]
    opt = torch.optim.Adam([p for n in nets for p in n.parameters()], lr=params.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=params.n_iters, eta_min=params.lr / 20)

    lam0 = max(0.1, lam_target)
    anneal_iters = max(1, int(0.7 * params.n_iters))

    def train_lambda(it):
        frac = min(1.0, it / anneal_iters)
        return lam0 * (lam_target / lam0) ** frac

    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    final_J = None
    for it in range(params.n_iters):
        lam_now = train_lambda(it)
        S_paths = gbm_paths(S0, params, n_paths=params.batch_size, device=device,
                            antithetic=True)   # halves pathwise-gradient variance
        frac = min(1.0, it / anneal_iters)
        smooth = 0.1 * (1e-3 / 0.1) ** frac              # 0.1 -> 1e-3, then held
        pay, ent, _, _ = _rollout(nets, S_paths, params, payoff_fn, device,
                                  lam_now, smooth=smooth)
        J = (pay + lam_now * ent).mean()
        opt.zero_grad()
        (-J).backward()                       # maximise J
        opt.step()
        sched.step()
        final_J = J.item()

    if device == "cuda":
        torch.cuda.synchronize()
    diag = {"final_objective_J": final_J, "device": device,
            "train_runtime": time.perf_counter() - t_start,
            "anneal": f"{lam0:g} -> {lam_target:g} over {anneal_iters} iters"}
    return nets, diag


def evaluate(nets, S0, params: Params, payoff_fn=payoffs.put, device=None):
    """Value the trained policy on fresh paths. Does not reseed; call after
    fit()."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lam_target = params.lambda_
    with torch.no_grad():
        S_test = gbm_paths(S0, params, n_paths=params.n_paths, device=device)
        pay, ent, otm, Y_T = _rollout(nets, S_test, params, payoff_fn, device,
                                      lam_target)
        M = pay.shape[0]
        value = pay.mean().item()
        stderr = (pay.std() / math.sqrt(M)).item()
        J_reg = (pay + lam_target * ent).mean().item()
    return {"value": value, "mc_stderr": stderr, "regularised_value_J": J_reg,
            "mean_terminal_fuel": Y_T.mean().item(),
            "otm_spend_frac": otm.mean().item()}


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
        value=ev["value"],                    # payoff-only, exact fuel weights
        runtime=runtime,
        diagnostics={
            "method": "dianetti",
            "device": device,
            "estimator": "pathwise policy gradient (not the paper's actor-critic)",
            "final_objective_J": tdiag["final_objective_J"],
            "regularised_value_J": ev["regularised_value_J"],
            "mc_stderr": ev["mc_stderr"],
            "greedy_policy_value": None,      # no greedy rule for a fuel policy
            "mean_terminal_fuel": ev["mean_terminal_fuel"],
            "otm_spend_frac": ev["otm_spend_frac"],
            "lambda_bias_scale": lam_target * params.T / math.e,
            "anneal": tdiag["anneal"],
        },
    )
