"""Least-squares Monte Carlo (LSM), Longstaff and Schwartz (2001).

- Backward recursion regressing realised discounted cash flows, not fitted
  values, on basis functions of the current state.
- In-the-money paths only in each regression.
- Prices normalised by the strike before entering the basis.

Basis: polynomials in the sorted normalised prices
y_1 >= y_2 >= ... >= y_d = sort(S/K):
- d = 1 : {1, x, x^2, x^3}
- d >= 2: {1} + {y_1^p, p = 1..5} + {y_i, y_i^2 for i = 2..d}
          + {y_i y_{i+1}} + the full product (d > 2).
This follows the paper's 19-function basis for the five-asset max-call, and
gives exactly 19 functions at d = 5.

price() values the executed policy on paths independent of the regression,
a low-biased estimate; the in-sample estimate is kept in diagnostics.
fit() returns an LSMPolicy for certification.
"""

import math
import time

import torch

from ..result import Result
from ..params import Params


# ----------------------------- basis functions ------------------------------

def _basis_size(dim):
    if dim == 1:
        return 4
    n = 1 + 5 + 2 * (dim - 1) + (dim - 1)      # const, max-powers, levels+squares, adjacents
    if dim > 2:
        n += 1                                  # full product
    return n


def _basis(S, K_strike, dim):
    """Design matrix (batch, B) from states (batch, dim); prices normalised by K."""
    y = S / K_strike
    if dim == 1:
        x = y[..., 0]
        return torch.stack([torch.ones_like(x), x, x ** 2, x ** 3], dim=-1)
    y = torch.sort(y, dim=-1, descending=True).values
    cols = [torch.ones_like(y[..., 0])]
    y1 = y[..., 0]
    cols += [y1 ** p for p in range(1, 6)]
    for i in range(1, dim):
        cols += [y[..., i], y[..., i] ** 2]
    cols += [y[..., i] * y[..., i + 1] for i in range(dim - 1)]
    if dim > 2:
        cols.append(torch.prod(y, dim=-1))
    return torch.stack(cols, dim=-1)


# ------------------------------ policy object -------------------------------

class LSMPolicy:
    """Exercise rule from fitted continuation values: exercise iff in the
    money and g >= C_fit. Dates with too few ITM training paths to regress
    hold no coefficients and never exercise (recorded in `skipped_dates`)."""

    def __init__(self, params: Params, payoff_fn, coeffs):
        self.K = params.K_strike
        self.dim = params.dim
        self.n_steps = params.n_steps
        self.payoff_fn = payoff_fn
        self.coeffs = coeffs                    # {k: float32 cpu tensor (B,)} for k=1..N-1
        self.skipped_dates = [k for k in range(1, params.n_steps) if k not in coeffs]
        self._device_cache = {}

    def _beta(self, k, device):
        key = (k, device)
        if key not in self._device_cache:
            self._device_cache[key] = self.coeffs[k].to(device)
        return self._device_cache[key]

    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        if k not in self.coeffs:
            return torch.zeros_like(g, dtype=torch.bool)
        C = _basis(S, self.K, self.dim) @ self._beta(k, S.device)
        return (g > 0.0) & (g >= C)


# ----------------------------- fit and price --------------------------------

def fit(S0, params: Params, payoff_fn=None, device="cpu", n_train=None,
        generator=None):
    """Run the LS backward recursion on a training path set; returns
    (LSMPolicy, in_sample_value, in_sample_stderr)."""
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
    B = _basis_size(params.dim)
    disc_step = math.exp(-params.r * dt)

    # cf: realised cash flow of the current (partial) stopping rule,
    # discounted to the date under consideration.
    g_N = payoff_fn(paths[:, N, :], params.K_strike)
    cf = g_N.clone()                            # tau = N to start
    coeffs = {}
    for k in range(N - 1, 0, -1):
        cf *= disc_step                         # discount one date, to date k
        S_k = paths[:, k, :]
        g_k = payoff_fn(S_k, params.K_strike)
        itm = g_k > 0.0
        if int(itm.sum()) < max(2 * B, 50):     # too few ITM paths: never exercise here
            continue
        X = _basis(S_k[itm], params.K_strike, params.dim).double()
        yv = cf[itm].double()
        beta = torch.linalg.lstsq(X, yv.unsqueeze(-1)).solution.squeeze(-1)
        coeffs[k] = beta.float().cpu()
        C = (X @ beta).float()
        ex = g_k[itm] >= C
        cf_itm = cf[itm]
        cf_itm[ex] = g_k[itm][ex]               # exercised paths: cash flow now
        cf[itm] = cf_itm

    in_sample = (cf * disc_step).clone()        # discount date 1 -> 0
    policy = LSMPolicy(params, payoff_fn, coeffs)
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
        diagnostics={
            "method": "lsm (out-of-sample policy value)",
            "mc_stderr": stderr,
            "in_sample_value": v_in, "in_sample_stderr": se_in,
            "n_basis": _basis_size(params.dim),
            "skipped_dates": policy.skipped_dates,
            "n_train": params.n_paths, "n_eval": params.n_paths,
        },
    )
