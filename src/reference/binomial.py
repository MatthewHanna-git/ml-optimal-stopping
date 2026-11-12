"""Binomial reference price for the single-asset Bermudan put.

A Cox-Ross-Rubinstein tree with `tree_substeps` steps between consecutive
exercise dates, where exercise is allowed only on the n_steps Bermudan
dates. The reference therefore prices the same Bermudan contract the
methods do, rather than the American one.
"""

import time
import numpy as np

from ..result import Result
from ..params import Params


def price(S0, params: Params) -> Result:
    """CRR tree, exercise restricted to the n_steps Bermudan dates."""
    t0 = time.perf_counter()

    L = params.n_steps                      # exercise dates
    sub = params.tree_substeps              # tree steps per exercise interval
    N = L * sub                             # total tree steps
    dt = params.T / N

    u = np.exp(params.sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-params.r * dt)
    p = (np.exp((params.r - params.delta) * dt) - d) / (u - d)

    # Terminal asset values S0 * u^j * d^(N-j), j = 0..N, and terminal payoff.
    j = np.arange(N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))
    V = np.maximum(params.K_strike - S_T, 0.0)

    # Backward induction. At step n there are n+1 nodes with S = S0 u^j d^(n-j).
    for n in range(N - 1, -1, -1):
        V = disc * (p * V[1:n + 2] + (1.0 - p) * V[0:n + 1])
        if n % sub == 0:                    # a Bermudan exercise date (includes t=0)
            S_n = S0 * (u ** np.arange(n + 1)) * (d ** (n - np.arange(n + 1)))
            V = np.maximum(V, params.K_strike - S_n)

    runtime = time.perf_counter() - t0
    return Result(
        value=float(V[0]),
        runtime=runtime,
        diagnostics={
            "method": "binomial (CRR, Bermudan-restricted)",
            "tree_steps": N,
            "exercise_dates": L,
        },
    )
