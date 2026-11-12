"""Exact simulation of d-dimensional geometric Brownian motion under the
risk-neutral measure.

    S^i_{t+dt} = S^i_t * exp((r - delta - sigma^2/2) dt + sigma sqrt(dt) Z^i)

with independent assets. Every Monte Carlo method draws its paths here.
Returns a tensor of shape (n_paths, n_steps + 1, dim), with index 0 the
starting price.
"""

import torch


def gbm_paths(S0, params, n_paths=None, device="cpu", generator=None,
              antithetic=False):
    """Simulate paths under the risk-neutral measure.

    S0 may be a scalar (same start for all assets) or a length-`dim` sequence.
    `generator` (a torch.Generator on the right device) makes runs reproducible.
    antithetic=True pairs each path with its mirrored driver (W, -W) — a
    variance-reduction option; n_paths should then be even.
    """
    n_paths = n_paths or params.n_paths
    d, L = params.dim, params.n_steps
    dt = params.T / L

    S0_t = torch.as_tensor(S0, dtype=torch.float32, device=device)
    if S0_t.ndim == 0:
        S0_t = S0_t.repeat(d)

    if antithetic:
        Z_half = torch.randn(n_paths // 2, L, d, device=device, generator=generator)
        Z = torch.cat([Z_half, -Z_half], dim=0)
    else:
        Z = torch.randn(n_paths, L, d, device=device, generator=generator)
    drift = (params.r - params.delta - 0.5 * params.sigma ** 2) * dt
    increments = drift + params.sigma * (dt ** 0.5) * Z

    log_paths = torch.cumsum(increments, dim=1)
    paths = torch.empty(n_paths, L + 1, d, device=device)
    paths[:, 0, :] = S0_t
    paths[:, 1:, :] = S0_t * torch.exp(log_paths)
    return paths
