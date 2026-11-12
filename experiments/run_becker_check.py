"""Validation of DOS under the Andersen-Broadie certification.

Each cell trains the per-date stopping networks and certifies the hard
policy. The anchors are Becker et al. (2019, Table 1), with their 2021
tables as secondary values.

(0) Budget: the two-asset cell at S0 = 100 is trained at each budget in
    SWEEP_BUDGETS, and the panels use the smallest whose policy value comes
    within SWEEP_TOL of the published lower bound (the largest if none
    does).
(a) The single-asset Bermudan put (50 dates), against the binomial value.
(b) The two- and three-asset max-call, against the published values and
    the two-asset tree.
(c) The five-asset max-call, against the published intervals and Broadie
    and Cao (2008).
(d) The ten-asset max-call, against the published intervals.

Each cell uses its own seed, 1000 d + S0.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import binomial, binomial2d
from src.methods import becker
from src.duality import certify
from src.duality.andersen_broadie import _policy_payoffs
from src import payoffs

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_SWEEP = True             # budget sweep first; sets STEPS_PER_DATE for the panels
RUN_PANEL_A = True
RUN_PANEL_B = True
RUN_PANEL_C = True
RUN_PANEL_D = True

SWEEP_BUDGETS = (400, 1500, 3000)   # steps/date candidates for the anchor cell
SWEEP_TOL = 0.02             # qualify if L >= 13.895 - TOL (2019 published L)
STEPS_PER_DATE = 400         # fallback / used directly when RUN_SWEEP = False
BATCH = 8192                 # paths per training step, as in the paper
LR = 1e-2                    # DOS learning rate (batch normalisation suits 1e-2)
N_EVAL = 2 ** 21             # lower-bound paths inside certify

AB_PUT = dict(n_outer=1500, n_inner=4000)
AB_MAXCALL = dict(n_outer=1500, n_inner=10000)

# Published anchors ----------------------------------------------------------
# Becker et al. 2019 Table 1 (symmetric case): L, U, 95% CI [+ binomial where given]
PUB_2019 = {
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
# Becker et al. 2021 Tables 9/10 means (secondary; joint-algorithm values)
PUB_2021 = {(2, 90.0): 8.068, (2, 100.0): 13.901, (2, 110.0): 21.341,
            (3, 90.0): 11.278, (3, 100.0): 18.685, (3, 110.0): 27.560,
            (5, 90.0): 16.631, (5, 100.0): 26.147, (5, 110.0): 36.774,
            (10, 90.0): 26.196, (10, 100.0): 38.272, (10, 110.0): 50.812}
# ----------------------------------------------------------------------------


def run_cell(S0, params, ref_value=None, pub=None, pub2021=None):
    t0 = time.perf_counter()
    policy, diag = becker.fit(S0, params, device=DEVICE)
    print(f"  dos fit           : {diag['steps_per_date']} steps/date x "
          f"{params.n_steps - 1} dates, batch {diag['batch_size']}, "
          f"{time.perf_counter() - t0:.0f}s")

    cert = certify(S0, params, policy, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  L0 (executed)     : {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})"
          f"   <- the DOS value")
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
    if pub2021 is not None:
        print(f"  published 2021    : {pub2021:.3f} (joint-algorithm mean)")


def maxcall_params(d, S0):
    N = 9
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=d, n_steps=N, tree_substeps=100, n_paths=N_EVAL,
                  n_iters=STEPS_PER_DATE * (N - 1), batch_size=BATCH,
                  lr=LR, seed=1000 * d + int(S0), **AB_MAXCALL)


def budget_sweep():
    """Train the d = 2, S0 = 100 cell at each candidate budget and keep the
    smallest whose policy value is within SWEEP_TOL of the published one."""
    global STEPS_PER_DATE
    print("=" * 72)
    print(f"(0) Budget sweep, d=2 S0=100 — rule: smallest steps/date with "
          f"L >= {13.895 - SWEEP_TOL:.3f}")
    print("=" * 72)
    chosen = None
    for spd in SWEEP_BUDGETS:
        N = 9
        params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                        dim=2, n_steps=N, n_iters=spd * (N - 1),
                        batch_size=BATCH, lr=LR, seed=2100, **AB_MAXCALL)
        t0 = time.perf_counter()
        policy, _ = becker.fit(100.0, params, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(2102)
        pay = _policy_payoffs(100.0, params, policy, payoffs.max_call,
                              N_EVAL, DEVICE, gen)
        L, se = pay.mean().item(), (pay.std() / (N_EVAL ** 0.5)).item()
        ok = L >= 13.895 - SWEEP_TOL
        print(f"  {spd:5d} steps/date: L = {L:8.4f} (stderr {se:.4f})   "
              f"{'QUALIFIES' if ok else 'short'}   ({time.perf_counter() - t0:.0f}s)")
        if ok and chosen is None:
            chosen = spd
    if chosen is None:
        chosen = SWEEP_BUDGETS[-1]
        print(f"\n  no budget reached {13.895 - SWEEP_TOL:.3f}; "
              f"panels run at the largest, {chosen} steps/date")
    else:
        print(f"\n  -> panels run at {chosen} steps/date")
    STEPS_PER_DATE = chosen


def panel_a():
    print("=" * 72)
    print("(a) 1-D Bermudan put (S0 = K = 40)")
    print("    anchors: binomial 5.311; Becker 2021 Table 4 d=1: 5.311 (0.004)")
    print("=" * 72)
    N = 50
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=N,
                    n_paths=N_EVAL, n_iters=STEPS_PER_DATE * (N - 1),
                    batch_size=BATCH, lr=LR, seed=1040, **AB_PUT)
    ref = binomial.price(40.0, params)
    print(f"  binomial reference: {ref.value:9.4f}")
    run_cell(40.0, params, ref_value=ref.value)


def panel_bcd(dims, title):
    print("=" * 72)
    print(title)
    print("=" * 72)
    for d in dims:
        for S0 in (90.0, 100.0, 110.0):
            params = maxcall_params(d, S0)
            print(f"\nd = {d}, S0 = {S0:.0f}")
            ref = None
            if d == 2:
                ref = binomial2d.price(S0, params).value
                print(f"  binomial2d ref    : {ref:9.4f}")
            run_cell(S0, params, ref_value=ref,
                     pub=PUB_2019.get((d, S0)), pub2021=PUB_2021.get((d, S0)))


def main():
    print(f"device: {DEVICE}")
    t0 = time.perf_counter()
    if RUN_SWEEP:
        budget_sweep()
    if RUN_PANEL_A:
        panel_a()
    if RUN_PANEL_B:
        panel_bcd([2, 3], "(b) d=2,3 max-call, Becker regime — vs 2019 Table 1 + binomial")
    if RUN_PANEL_C:
        panel_bcd([5], "(c) d=5 max-call — vs 2019 Table 1 + Broadie-Cao")
    if RUN_PANEL_D:
        panel_bcd([10], "(d) d=10 max-call — the pre-sweep depth anchor")
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
