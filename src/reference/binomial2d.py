"""Binomial reference price for the Bermudan max-call on two independent assets.

A product Cox-Ross-Rubinstein lattice: each asset moves up by
u = e^{sigma sqrt(dt)} or down by 1/u independently at each step, with
p = (e^{(r - delta) dt} - d) / (u - d). As in binomial.py, exercise is
allowed only on the n_steps Bermudan dates, with tree_substeps steps between
them.

Work and memory grow as O(N^2) per step in the total steps N, so choose
tree_substeps with care: 100 dates at 10 substeps takes tens of seconds.
"""

import time
import numpy as np

from ..result import Result
from ..params import Params


def price(S0, params: Params, payoff_fn=None) -> Result:
    """Product-CRR tree for dim=2, exercise restricted to the Bermudan dates.

    payoff_fn: operates on an array of shape (..., 2); defaults to the
    max-call at params.K_strike."""
    t0 = time.perf_counter()
    assert params.dim == 2, "binomial2d is the dim=2 reference"

    if payoff_fn is None:
        from .. import payoffs
        payoff_fn = payoffs.max_call

    L = params.n_steps
    sub = params.tree_substeps
    N = L * sub
    dt = params.T / N

    u = np.exp(params.sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-params.r * dt)
    p = (np.exp((params.r - params.delta) * dt) - d) / (u - d)
    q = 1.0 - p
    pp, pq, qq = p * p, p * q, q * q

    def payoff_grid(n):
        """Payoff on the step-n lattice: S_i = S0 * u^{2j - n}, j = 0..n."""
        s = S0 * u ** (2.0 * np.arange(n + 1) - n)          # (n+1,)
        S = np.stack(np.meshgrid(s, s, indexing="ij"), axis=-1)  # (n+1, n+1, 2)
        return payoff_fn(S, params.K_strike)

    V = payoff_grid(N)
    for n in range(N - 1, -1, -1):
        V = disc * (pp * V[1:, 1:] + pq * (V[1:, :-1] + V[:-1, 1:]) + qq * V[:-1, :-1])
        if n % sub == 0:                                    # Bermudan date (incl. t=0)
            V = np.maximum(V, payoff_grid(n))

    runtime = time.perf_counter() - t0
    return Result(
        value=float(V[0, 0]),
        runtime=runtime,
        diagnostics={
            "method": "binomial2d (product CRR, Bermudan-restricted)",
            "tree_steps": N,
            "exercise_dates": L,
        },
    )
