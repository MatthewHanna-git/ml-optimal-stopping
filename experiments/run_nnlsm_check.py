"""Validation of NN-LSM under the Andersen-Broadie certification, with a budget
check on RDBDP.

(a) The single-asset Bermudan put, against the binomial value.
(b) The two-asset max-call, against the tree and the published values.
(c) The five-asset max-call, against the published intervals and Lapeyre
    and Lelong (2021, Table 9), whose cell this is at S0 = 100.
(d) The ten-asset max-call at S0 = 100, at 1500 steps per date.
(e) RDBDP at d = 5, S0 = 110 refitted at 1500 steps per date: the larger
    budget is adopted for d >= 5 if it halves the gap at 400 steps per date
    (HURE_GAP_TARGET).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import binomial, binomial2d
from src.methods import nn_lsm, hure
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_PANEL_A = True
RUN_PANEL_B = True
RUN_PANEL_C = True
RUN_PANEL_D = True
RUN_PANEL_E = True

STEPS_PER_DATE = 400
STEPS_PER_DATE_D10 = 1500
N_TRAIN = 2 ** 18            # one training path set, reused across dates
BATCH = 8192
LR = 1e-2
N_EVAL = 2 ** 21

AB_MAXCALL = dict(n_outer=1500, n_inner=10000)
AB_PUT = dict(n_outer=1500, n_inner=4000)

HURE_GAP_400 = 0.3231        # RDBDP's gap at d = 5, S0 = 110, 400 steps per date
HURE_GAP_TARGET = 0.16       # adopt 1500/date iff gap <= this

PUB_2019 = {
    (2, 90.0): dict(L=8.072, U=8.075, ci=(8.060, 8.081), binomial=8.075),
    (2, 100.0): dict(L=13.895, U=13.903, ci=(13.880, 13.910), binomial=13.902),
    (2, 110.0): dict(L=21.353, U=21.346, ci=(21.336, 21.354), binomial=21.345),
    (5, 90.0): dict(L=16.648, U=16.640, ci=(16.633, 16.648), bc_ci=(16.620, 16.653)),
    (5, 100.0): dict(L=26.156, U=26.162, ci=(26.138, 26.174), bc_ci=(26.115, 26.164)),
    (5, 110.0): dict(L=36.766, U=36.777, ci=(36.745, 36.789), bc_ci=(36.710, 36.798)),
    (10, 100.0): dict(L=38.321, U=38.353, ci=(38.300, 38.367)),
}
LAPEYRE_NOTE = {  # their Table 9 (d=5 S0=100 cell only)
    (5, 100.0): "Lapeyre T9: NN-LSM 25.92-26.02 (+-~0.1), LS-poly3 25.98, poly6 26.29",
}
# ----------------------------------------------------------------------------


def run_cell(S0, params, ref_value=None, pub=None, note=None):
    t0 = time.perf_counter()
    policy, v_in, se_in = nn_lsm.fit(S0, params, device=DEVICE,
                                     n_train=N_TRAIN)
    print(f"  nn-lsm fit        : {params.n_iters // (params.n_steps - 1)} "
          f"steps/date, {time.perf_counter() - t0:.0f}s"
          + (f", skipped dates {policy.skipped_dates}"
             if policy.skipped_dates else ""))
    print(f"  in-sample (paper) : {v_in:9.4f}  (stderr {se_in:.4f}; "
          f"their estimator convention, not a bound)")

    cert = certify(S0, params, policy, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  L0 (out-of-sample): {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})"
          f"   <- the NN-LSM value")
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
    if note is not None:
        print(f"  {note}")


def maxcall_params(d, S0, spd):
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=d, n_steps=9, tree_substeps=100, n_paths=N_EVAL,
                  n_iters=spd * 8, batch_size=BATCH, lr=LR,
                  seed=9000 + 100 * d + int(S0), **AB_MAXCALL)


def panel_a():
    print("=" * 72)
    print("(a) 1-D Bermudan put (S0 = K = 40) — anchor: binomial 5.311")
    print("=" * 72)
    N = 50
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=N,
                    n_paths=N_EVAL, n_iters=STEPS_PER_DATE * (N - 1),
                    batch_size=BATCH, lr=LR, seed=9040, **AB_PUT)
    ref = binomial.price(40.0, params)
    print(f"  binomial reference: {ref.value:9.4f}")
    run_cell(40.0, params, ref_value=ref.value)


def panel_bcd(dims, spd, title):
    print("=" * 72)
    print(title)
    print("=" * 72)
    for d in dims:
        for S0 in ((100.0,) if d == 10 else (90.0, 100.0, 110.0)):
            params = maxcall_params(d, S0, spd)
            print(f"\nd = {d}, S0 = {S0:.0f}")
            ref = None
            if d == 2:
                ref = binomial2d.price(S0, params).value
                print(f"  binomial2d ref    : {ref:9.4f}")
            run_cell(S0, params, ref_value=ref, pub=PUB_2019.get((d, S0)),
                     note=LAPEYRE_NOTE.get((d, S0)))


def panel_e():
    print("=" * 72)
    print(f"(e) HURE BUDGET PROBE: d=5 S0=110 at 1500/date")
    print(f"    gap was {HURE_GAP_400} at 400/date; rule: adopt 1500/date "
          f"iff gap <= {HURE_GAP_TARGET}")
    print("=" * 72)
    params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                    dim=5, n_steps=9, n_paths=N_EVAL, n_iters=1500 * 8,
                    batch_size=BATCH, lr=LR, seed=9511, **AB_MAXCALL)
    t0 = time.perf_counter()
    policy, _ = hure.fit(110.0, params, device=DEVICE)
    print(f"  rdbdp fit         : 1500 steps/date, {time.perf_counter() - t0:.0f}s")
    cert = certify(110.0, params, policy, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  L0 {c['lower']:9.4f} ({c['lower_stderr']:.4f})   "
          f"U0 {c['upper']:9.4f}   CI [{lo:.4f}, {hi:.4f}]   gap {c['gap']:.4f}")
    if c["gap"] <= HURE_GAP_TARGET:
        print(f"  -> gap halved: adopt 1500 steps/date for RDBDP at d >= 5")
    else:
        print(f"  -> gap did not halve: keep 400 steps/date")


def main():
    print(f"device: {DEVICE}")
    t0 = time.perf_counter()
    if RUN_PANEL_A:
        panel_a()
    if RUN_PANEL_B:
        panel_bcd([2], STEPS_PER_DATE, "(b) d=2 max-call — vs tree + 2019 Table 1")
    if RUN_PANEL_C:
        panel_bcd([5], STEPS_PER_DATE, "(c) d=5 max-call — THE LAPEYRE CELL")
    if RUN_PANEL_D:
        panel_bcd([10], STEPS_PER_DATE_D10, "(d) d=10 S0=100 — depth spot check")
    if RUN_PANEL_E:
        panel_e()
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
