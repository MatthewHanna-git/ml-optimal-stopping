"""The Params bundle shared by every method.

Market parameters, discretisation and method settings live in one object, so
an experiment builds a single Params and passes it to every method. A method
ignores the fields it does not use.

Naming: `lambda_` avoids the Python keyword; `K_strike` is the strike, kept
apart from the Dai/Ye penalty (also called K in those papers), which is
`penalty`; `delta` is the continuous dividend yield.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Params:
    # --- market / contract ---
    K_strike: float
    r: float
    sigma: float
    T: float
    delta: float = 0.0
    dim: int = 1

    # --- discretisation ---
    n_steps: int = 50          # number of exercise dates (Bermudan grid)
    n_paths: int = 2 ** 16     # Monte Carlo paths for training/evaluation

    # --- method dials ---
    lambda_: float = 1e-3      # temperature of the RL methods
    penalty: Optional[float] = None  # K/N penalty for Dai/Ye; None where absent
    # Starting temperature of the annealing in Dong and Chee. None anneals
    # geometrically from max(0.1, lambda_) to lambda_ over the first 70% of
    # iterations; lambda_start == lambda_ holds the temperature constant.
    lambda_start: Optional[float] = None

    # --- network / training ---
    arch: Tuple[int, ...] = (50, 50)  # hidden-layer widths of the shared MLP
    n_iters: int = 3000        # outer training iterations
    batch_size: int = 1024     # paths per training iteration
    lr: float = 1e-3           # initial Adam learning rate (cosine-decayed during training)

    # --- reference-method knobs ---
    tree_substeps: int = 40    # binomial tree steps *per exercise date*

    # --- duality (Andersen-Broadie) budgets ---
    n_outer: int = 1500        # N1: outer dual paths (AB 2004 Table 2 setting)
    n_inner: int = 10000       # N2 = N3: inner subpaths per date (AB Table 2 setting)

    # --- reproducibility ---
    seed: Optional[int] = 0
