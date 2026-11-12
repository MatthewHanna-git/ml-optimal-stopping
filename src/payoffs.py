"""Payoff functions for torch tensors and numpy arrays.

Both take prices of shape (..., d), with assets on the last axis, and return
shape (...).
- put:      the single-asset American put.
- max_call: the call on the maximum of d assets.
"""

import numpy as np

try:
    import torch
except ImportError:          # numpy-only use, e.g. the binomial trees
    torch = None


def _clip0(x):
    """max(x, 0) for torch tensors or numpy arrays."""
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.clamp(x, min=0.0)
    return np.maximum(x, 0.0)


def _last_axis_max(S):
    """max over the asset axis for torch tensors or numpy arrays."""
    if torch is not None and isinstance(S, torch.Tensor):
        return S.max(dim=-1).values
    return np.max(S, axis=-1)


def put(S, K_strike):
    """(K - S)^+ for a single asset. S has shape (..., 1) or (...,)."""
    S1 = S[..., 0] if S.ndim > 1 else S
    return _clip0(K_strike - S1)


def max_call(S, K_strike):
    """(max_i S_i - K)^+ over the last axis. S has shape (..., d)."""
    return _clip0(_last_axis_max(S) - K_strike)
