"""Validation of RDBDP under the Andersen-Broadie certification.

(a) The single-asset Bermudan put (50 dates), against the binomial value.
(b) The geometric put of Huré et al. (Section 5.3, Table 11): five assets,
    payoff (K - prod_i X_i)^+, K = S0 = 1, r = 5%, sigma = 20%, T = 1, 40
    dates. The product of the assets is itself a GBM, so the single-asset
    tree prices the Bermudan contract exactly, with delta = r - d r. Their
    reference, 0.10738, is the American value.
(c) The two-, three- and five-asset max-call at 400 steps per date, against
    Becker et al. (2019, Table 1), the trees and Broadie and Cao (2008).
(d) The ten-asset max-call at 1500 steps per date.

Each cell prints the certified L0 beside the scheme's own estimate U_0.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import binomial, binomial2d
from src.methods import hure
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_PANEL_A = True
RUN_PANEL_B = True
RUN_PANEL_C = True
RUN_PANEL_D = True

STEPS_PER_DATE = 400         # budget for d <= 5
STEPS_PER_DATE_D10 = 1500    # budget for d >= 10
BATCH = 8192
LR = 1e-2
N_EVAL = 2 ** 21

AB_MAXCALL = dict(n_outer=1500, n_inner=10000)   # 9-date panels
AB_LONG = dict(n_outer=1500, n_inner=2000)       # 40/50-date panels

PUB_2019 = {  # Becker 2019 Table 1 anchors for panel (c)/(d)
    (2, 90.0): dict(L=8.072, U=8.075, ci=(8.060, 8.081), binomial=8.075),
    (2, 100.0): dict(L=13.895, U=13.903, ci=(13.880, 13.910), binomial=13.902),
    (2, 110.0): dict(L=21.353, U=21.346, ci=(21.336, 21.354), binomial=21.345),
    (3, 90.0): dict(L=11.290, U=11.283, ci=(11.276, 11.290), binomial=11.29),
    (3, 100.0): dict(L=18.690, U=18.691, ci=(18.673, 18.699), binomial=18.69),
    (3, 110.0): dict(L=27.564, U=27.581, ci=(27.545, 27.591), binomial=27.58),
    (5, 90.0): dict(L=16.648, U=16.640, ci=(16.633, 16.648), bc_ci=(16.620, 16.653)),
    (5, 100.0): dict(L=26.156, U=26.162, ci=(26.138, 26.174), bc_ci=(26.115, 26.164)),
    (5, 110.0): dict(L=36.766, U=36.777, ci=(36.745, 36.789), bc_ci=(36.710, 36.798)),
    (10, 90.0): dict(L=26.208, U=26.272, ci=(26.189, 26.289)),
    (10, 100.0): dict(L=38.321, U=38.353, ci=(38.300, 38.367)),
    (10, 110.0): dict(L=50.857, U=50.914, ci=(50.834, 50.937)),
}
# ----------------------------------------------------------------------------


def geometric_put(S, K_strike):
    """(K - prod_i S_i)+ — Hure Section 5.3's payoff. Torch/numpy agnostic."""
    prod = S.prod(dim=-1) if isinstance(S, torch.Tensor) else S.prod(axis=-1)
    z = prod * 0.0
    return torch.maximum(K_strike - prod, z) if isinstance(S, torch.Tensor) \
        else __import__("numpy").maximum(K_strike - prod, 0.0)


def run_cell(S0, params, payoff_fn=None, ref_value=None, pub=None):
    t0 = time.perf_counter()
    policy, diag = hure.fit(S0, params, payoff_fn=payoff_fn, device=DEVICE)
    print(f"  rdbdp fit         : {diag['steps_per_date']} steps/date x "
          f"{params.n_steps - 1} dates, {time.perf_counter() - t0:.0f}s")

    cert = certify(S0, params, policy, payoff_fn=payoff_fn, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  L0 (executed)     : {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})"
          f"   <- the RDBDP value")
    print(f"  A-B upper U0      : {c['upper']:9.4f}  (stderr {c['delta_stderr']:.4f}"
          f" on delta {c['delta']:.4f})")
    print(f"  95% CI            : [{lo:.4f}, {hi:.4f}]   gap {c['gap']:.4f}   "
          f"({cert.runtime:.1f}s)")
    if ref_value is not None:
        inside = lo <= ref_value <= hi
        print(f"  reference         : {ref_value:9.4f}   "
              f"{'INSIDE CI' if inside else '*** OUTSIDE CI ***'}")
    if pub is not None:
        line = (f"  published 2019    : L {pub['L']:.3f}   U {pub['U']:.3f}"
                f"   CI [{pub['ci'][0]:.3f}, {pub['ci'][1]:.3f}]")
        if "binomial" in pub:
            line += f"   binomial {pub['binomial']:.3f}"
        if "bc_ci" in pub:
            line += f"   BC CI [{pub['bc_ci'][0]:.3f}, {pub['bc_ci'][1]:.3f}]"
        print(line)


def panel_a():
    print("=" * 72)
    print("(a) 1-D Bermudan put (S0 = K = 40) — anchor: binomial 5.311")
    print("=" * 72)
    N = 50
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=N,
                    n_paths=N_EVAL, n_iters=STEPS_PER_DATE * (N - 1),
                    batch_size=BATCH, lr=LR, seed=8040, **AB_LONG)
    ref = binomial.price(40.0, params)
    print(f"  binomial reference: {ref.value:9.4f}")
    run_cell(40.0, params, ref_value=ref.value)


def panel_b():
    print("=" * 72)
    print("(b) Geometric put d=5 — Hure's own Table 11 anchor")
    print("    reference: the Bermudan-40 tree below. Their N=40 estimate is")
    print("    0.10725 (0.00012); their 'reference' 0.10738 is the AMERICAN")
    print("    limit (their N=80 value 0.10765 overshoots it — not a bound)")
    print("=" * 72)
    d, N = 5, 40
    # exact reference: the product asset on the single-asset tree
    ref_params = Params(K_strike=1.0, r=0.05, sigma=0.2 * d ** 0.5, T=1.0,
                        delta=0.05 - d * 0.05, dim=1, n_steps=N,
                        tree_substeps=50)
    ref = binomial.price(1.0, ref_params)
    print(f"  binomial reference: {ref.value:9.5f}   (1-D product-asset "
          f"reduction, Bermudan-{N})")
    params = Params(K_strike=1.0, r=0.05, sigma=0.2, T=1.0, delta=0.0, dim=d,
                    n_steps=N, n_paths=N_EVAL,
                    n_iters=STEPS_PER_DATE * (N - 1), batch_size=BATCH,
                    lr=LR, seed=8540, **AB_LONG)
    run_cell(1.0, params, payoff_fn=geometric_put, ref_value=ref.value)


def panel_cd(dims, spd, title):
    print("=" * 72)
    print(title)
    print("=" * 72)
    for d in dims:
        for S0 in (90.0, 100.0, 110.0):
            params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0,
                            delta=0.1, dim=d, n_steps=9, tree_substeps=100,
                            n_paths=N_EVAL, n_iters=spd * 8, batch_size=BATCH,
                            lr=LR, seed=8000 + 100 * d + int(S0), **AB_MAXCALL)
            print(f"\nd = {d}, S0 = {S0:.0f}")
            ref = None
            if d == 2:
                ref = binomial2d.price(S0, params).value
                print(f"  binomial2d ref    : {ref:9.4f}")
            run_cell(S0, params, ref_value=ref, pub=PUB_2019.get((d, S0)))


def main():
    print(f"device: {DEVICE}")
    t0 = time.perf_counter()
    if RUN_PANEL_A:
        panel_a()
    if RUN_PANEL_B:
        panel_b()
    if RUN_PANEL_C:
        panel_cd([2, 3, 5], STEPS_PER_DATE,
                 "(c) d=2,3,5 max-call — vs 2019 Table 1 + binomials + BC")
    if RUN_PANEL_D:
        panel_cd([10], STEPS_PER_DATE_D10,
                 "(d) d=10 max-call — adopted 1500/date budget")
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
