"""The Policy interface, and the binomial-tree policies used to validate the
certification.

A policy is anything with
    policy.exercise(k, S) -> bool tensor of shape (batch,)
for a date index k in {1, ..., n_steps} and states S of shape (batch, dim).
This exercise indicator is all Andersen-Broadie needs, so every method is
certified through a small adapter that implements it. certify() applies the
g > 0 guard, so policies need not.

The tree policies take the near-optimal exercise rule implied by the
binomial references on the Bermudan grid. They let the certification be
checked against a known price. The tree construction is repeated here
rather than shared with src/reference/, so the references stay untouched.
"""

from typing import Protocol

import numpy as np
import torch

from ..params import Params


class Policy(Protocol):
    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        """Exercise indicator at date k for states S: (batch, dim) -> (batch,) bool."""
        ...


class PutTreePolicy1D:
    """Exercise boundary b_k extracted from the 1-D CRR tree (put: S <= b_k).

    At each Bermudan date the tree's nodes split into an exercise region
    (payoff >= continuation) below the boundary and a continuation region
    above it; b_k is the geometric midpoint between the last exercise node
    and the first continuation node. Boundary resolution is one node
    spacing sigma*sqrt(dt_tree) in log-price — with tree_substeps fine
    steps, a fraction of a percent.
    """

    def __init__(self, S0: float, params: Params):
        assert params.dim == 1, "PutTreePolicy1D is the dim=1 policy"
        L, sub = params.n_steps, params.tree_substeps
        N = L * sub
        dt = params.T / N
        u = np.exp(params.sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-params.r * dt)
        p = (np.exp((params.r - params.delta) * dt) - d) / (u - d)

        j = np.arange(N + 1)
        S_T = S0 * (u ** j) * (d ** (N - j))
        V = np.maximum(params.K_strike - S_T, 0.0)

        boundary = np.zeros(L + 1)              # b_k for k = 1..L (index 0 unused)
        for n in range(N - 1, -1, -1):
            V = disc * (p * V[1:n + 2] + (1.0 - p) * V[0:n + 1])
            if n % sub == 0:
                S_n = S0 * (u ** np.arange(n + 1)) * (d ** (n - np.arange(n + 1)))
                h = np.maximum(params.K_strike - S_n, 0.0)
                if n > 0:
                    ex = (h >= V) & (h > 0.0)   # exercise region: low-S nodes
                    if ex.any():
                        i_last = int(np.max(np.nonzero(ex)[0]))
                        b = (np.sqrt(S_n[i_last] * S_n[i_last + 1])
                             if i_last + 1 <= n else S_n[i_last])
                        boundary[n // sub] = b
                V = np.maximum(V, h)

        self.boundary = boundary                # numpy, b_k = 0 means never exercise
        self.tree_value = float(V[0])           # cross-check against binomial.price

    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        return S[..., 0] <= float(self.boundary[k])


class MaxCallTreePolicy2D:
    """Exercise rule implied by the two-asset product tree (max-call).

    The exercise region is a surface, not a threshold, so the rule stores
    the tree's continuation values C_k(S1, S2) at each Bermudan date and
    exercises iff payoff >= C, interpolated bilinearly in log-price. States
    off the lattice are clamped to its edge, where the region is clear.
    """

    def __init__(self, S0: float, params: Params, payoff_fn=None):
        assert params.dim == 2, "MaxCallTreePolicy2D is the dim=2 policy"
        if payoff_fn is None:
            from .. import payoffs
            payoff_fn = payoffs.max_call
        self.payoff_fn = payoff_fn
        self.K = params.K_strike

        L, sub = params.n_steps, params.tree_substeps
        N = L * sub
        dt = params.T / N
        u = np.exp(params.sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-params.r * dt)
        p = (np.exp((params.r - params.delta) * dt) - d) / (u - d)
        q = 1.0 - p
        pp, pq, qq = p * p, p * q, q * q
        log_u = np.log(u)
        log_S0 = np.log(S0)

        def payoff_grid(n):
            s = S0 * u ** (2.0 * np.arange(n + 1) - n)
            S = np.stack(np.meshgrid(s, s, indexing="ij"), axis=-1)
            return payoff_fn(S, params.K_strike)

        V = payoff_grid(N)
        # Per date k = 1..L-1: (x0, step, C-grid); log-prices x_j = x0 + j*step.
        self._grids = {}
        for n in range(N - 1, -1, -1):
            V = disc * (pp * V[1:, 1:] + pq * (V[1:, :-1] + V[:-1, 1:])
                        + qq * V[:-1, :-1])
            if n % sub == 0:
                k = n // sub
                if 0 < k < L:
                    x0 = log_S0 - n * log_u
                    self._grids[k] = (x0, 2.0 * log_u,
                                      torch.from_numpy(V.astype(np.float32)))
                V = np.maximum(V, payoff_grid(n))

        self.n_steps = L
        self.tree_value = float(V[0, 0])        # cross-check against binomial2d.price
        self._device_cache = {}

    def _cont(self, k: int, S: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of the date-k continuation grid at states S."""
        x0, step, C_cpu = self._grids[k]
        key = (k, S.device)
        if key not in self._device_cache:
            self._device_cache[key] = C_cpu.to(S.device)
        C = self._device_cache[key]
        n_max = C.shape[0] - 1

        f = (torch.log(S) - x0) / step                       # fractional indices
        f = f.clamp(0.0, float(n_max))
        i0 = f.floor().long().clamp(max=n_max - 1)
        w = f - i0.float()
        i1, j1 = i0[..., 0], i0[..., 1]
        w1, w2 = w[..., 0], w[..., 1]
        c = (C[i1, j1] * (1 - w1) * (1 - w2) + C[i1 + 1, j1] * w1 * (1 - w2)
             + C[i1, j1 + 1] * (1 - w1) * w2 + C[i1 + 1, j1 + 1] * w1 * w2)
        return c

    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:                   # maturity: exercise iff in the money
            return g > 0.0
        return g >= self._cont(k, S)
