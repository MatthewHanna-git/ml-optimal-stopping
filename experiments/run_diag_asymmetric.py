"""Diagnostic: the structure test on asymmetric assets.

On the symmetric max-call, sorting the prices is exact knowledge: the value
depends on the set of prices only. With a different volatility for each asset
it no longer is: which asset holds a price matters, and sorting discards it.
This run repeats the two-by-two of the structure test on the asymmetric
max-call of Becker et al. (2019, Table 2),

    sigma_i = 0.08 + 0.32 (i - 1)/(d - 1)  for d <= 5,
    sigma_i = 0.1 + i/(2d)                 for d > 5,

with every other parameter as in the sweep (K = 100, r = 5%, delta = 10%,
independent assets, T = 3, 9 dates). Six variants per cell:

    lsm             the sweep's sorted basis (now partly wrong knowledge)
    lsm_generic     degree-2 polynomials in (S/K, g/K), unsorted
    nn_lsm, becker  as in the sweep (prices in asset order, with g)
    nn_lsm_sorted, becker_sorted   prices sorted before the network

Budgets, training sizes and certification are the sweep's, held, not
recalibrated for this harder contract; the comparisons are paired. All six
variants of a cell share one seed: the same training paths within each
method and the same certification paths across all six.

The run has two phases:

  1. Validation, d in {2, 3}, S0 in {90, 100, 110}: every interval is set
     against Becker et al.'s Table 2 interval, and the run stops if one of
     the sweep's own variants (lsm, nn_lsm, becker) fails to overlap it.
  2. Main, d in {5, 10, 20, 50}, S0 in {90, 100, 110}.

Paired contrasts are printed for each cell and read only beyond 2 combined
standard errors.

Writes results/diag_asymmetric.csv (resumable) and the trained policies to
results/nets_diag/. About two and a half hours on an RTX 3080.

Usage:  python experiments/run_diag_asymmetric.py
        python experiments/run_diag_asymmetric.py --smoke
"""

import csv
import dataclasses
import math
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch

import run_sweep as sweep
import run_diag_generic_lsm as generic
from src import payoffs
from src.reference import lsm
from src.methods import nn_lsm, becker
from src.methods.common import ValueNet, StopNet
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_asymmetric.csv" if SMOKE else "diag_asymmetric.csv")
NETS_DIR = RESULTS / ("nets_diag_smoke" if SMOKE else "nets_diag")

VARIANTS = ["lsm", "lsm_generic", "nn_lsm", "nn_lsm_sorted", "becker", "becker_sorted"]
VALIDATION = [(d, s) for d in (2, 3) for s in (90.0, 100.0, 110.0)]
# S0 = 100 first, then 90 and 110: the order fixes each cell's index in CELLS,
# and so its seed.
MAIN = [(d, 100.0) for d in (5, 10, 20, 50)]
MAIN += [(d, s) for d in (5, 10, 20, 50) for s in (90.0, 110.0)]
if SMOKE:
    VALIDATION = [(2, 100.0)]
    MAIN = [(5, 100.0)]
CELLS = VALIDATION + MAIN
SEED_BASE, SEED_STEP = 5_000_000, 10_000     # above every other block in the repo
READ_Z = 2.0

# Becker, Cheridito and Jentzen (2019), Table 2: 95% CI (asymmetric assets)
BECKER_T2 = {
    (2, 90.0): (14.299, 14.367), (2, 100.0): (19.772, 19.829), (2, 110.0): (27.138, 27.163),
    (3, 90.0): (19.065, 19.104), (3, 100.0): (26.648, 26.701), (3, 110.0): (35.806, 35.835),
    (5, 90.0): (27.630, 27.680), (5, 100.0): (37.940, 38.014), (5, 110.0): (49.445, 49.533),
    (10, 90.0): (85.857, 86.087), (10, 100.0): (104.603, 104.864), (10, 110.0): (123.570, 123.904),
    (20, 90.0): (125.819, 126.383), (20, 100.0): (149.480, 150.053), (20, 110.0): (173.144, 173.937),
    (50, 90.0): (195.793, 196.963), (50, 100.0): (227.247, 228.605), (50, 110.0): (258.661, 260.092),
}

FIELDS = ["phase", "variant", "d", "S0", "seed", "budget",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "becker_lo", "becker_hi", "overlaps_becker", "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def asym_sigma(d):
    i = torch.arange(1, d + 1, dtype=torch.float32)
    s = 0.08 + 0.32 * (i - 1) / (d - 1) if d <= 5 else 0.1 + i / (2 * d)
    return s.to(DEVICE)


def cell_seed(d, S0):
    return SEED_BASE + SEED_STEP * CELLS.index((d, S0))


def asym_params(d, seed, extra):
    p = sweep.make_params(d, seed, extra)
    return dataclasses.replace(p, sigma=asym_sigma(d))


def _desc(S):
    return torch.sort(S, dim=-1, descending=True).values


class SortedValueNet(ValueNet):
    def forward(self, S):
        return super().forward(_desc(S))


class SortedStopNet(StopNet):
    def forward(self, S):
        return super().forward(_desc(S))


def fit_variant(variant, d, S0, seed):
    """Train one variant as the sweep would. Returns (params, policy, budget, fit_s, state)."""
    torch.manual_seed(seed)
    if variant in ("lsm", "lsm_generic"):
        params = asym_params(d, seed, {})
        sweep._sync(); t0 = time.perf_counter()
        if variant == "lsm":
            policy, _, _ = lsm.fit(S0, params, device=DEVICE, n_train=sweep.N_TRAIN)
        else:
            basis = generic.GenericBasis(d, 2, True, params.K_strike)
            policy, _ = generic.fit(S0, params, basis, sweep.N_TRAIN)
        state = {"coeffs": {int(k): v for k, v in policy.coeffs.items()}}
        sweep._sync()
        return params, policy, f"n_train=2^{sweep.N_TRAIN.bit_length() - 1}", \
            time.perf_counter() - t0, state
    method = variant.replace("_sorted", "")
    sorted_ = variant.endswith("_sorted")
    nn_lsm.ValueNet = SortedValueNet if sorted_ else ValueNet      # fit() builds by these names
    becker.StopNet = SortedStopNet if sorted_ else StopNet
    spd = sweep.SPD[method][d]
    params = asym_params(d, seed, dict(n_iters=spd * 8, batch_size=sweep.BATCH_ML, lr=sweep.LR_ML))
    sweep._sync(); t0 = time.perf_counter()
    if method == "nn_lsm":
        policy, _, _ = nn_lsm.fit(S0, params, device=DEVICE, n_train=sweep.N_TRAIN)
    else:
        policy, _ = becker.fit(S0, params, device=DEVICE)
    sweep._sync()
    return params, policy, f"{spd}/date", time.perf_counter() - t0, {"nets": sweep._state(policy.nets)}


def load_done():
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, newline="") as f:
        return {(r["variant"], int(r["d"]), float(r["S0"])): r for r in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def compare(rows, d, S0):
    """Paired contrasts within a cell, printed after its six variants."""
    g = {v: rows.get((v, d, S0)) for v in VARIANTS}
    if not all(g.values()):
        return
    L = {v: float(r["L0"]) for v, r in g.items()}
    se = {v: float(r["L0_stderr"]) for v, r in g.items()}

    def line(label, a, b):
        diff = L[a] - L[b]
        z = diff / math.sqrt(se[a] ** 2 + se[b] ** 2)
        print(f"    {label:28s} {diff:+.4f} (z={z:+.1f}{', read' if abs(z) > READ_Z else ''})")

    print(f"  contrasts d={d} S0={S0:.0f}:")
    line("LSM sorted basis - generic", "lsm", "lsm_generic")
    line("NN-LSM sorted - unsorted", "nn_lsm_sorted", "nn_lsm")
    line("DOS sorted - unsorted", "becker_sorted", "becker")
    line("LSM - DOS (as in the sweep)", "lsm", "becker")
    line("DOS sorted - LSM", "becker_sorted", "lsm")


def main():
    sweep.env_banner()
    print(f"diag csv: {CSV_PATH}\ndiag nets: {NETS_DIR}")
    print("contract: asymmetric max-call, Becker et al. (2019) Table 2")
    seeds = [cell_seed(d, s) for d, s in CELLS]
    assert len(set(seeds)) == len(seeds) and min(seeds) > 3_500_000
    rows = load_done()
    if rows:
        print(f"resuming: {len(rows)} runs already recorded, will be skipped")
    t_all = time.perf_counter()

    for phase, cells in (("validation", VALIDATION), ("main", MAIN)):
        print(f"\n===== {phase.upper()} =====")
        for d, S0 in cells:
            sig = asym_sigma(d)
            print(f"-- d={d} S0={S0:.0f}  sigma {sig.min().item():.3f}..{sig.max().item():.3f}  "
                  f"Becker T2 CI {BECKER_T2[(d, S0)]}")
            for variant in VARIANTS:
                if (variant, d, S0) in rows:
                    print(f"  skip (done): {variant}")
                    continue
                seed = cell_seed(d, S0)
                params, policy, budget, fit_s, state = fit_variant(variant, d, S0, seed)
                NETS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"variant": variant, "d": d, "S0": S0, "seed": seed,
                            "sigma": sig.cpu().tolist(), "budget": budget, **state},
                           NETS_DIR / f"asym_{variant}_d{d}_S{S0:.0f}.pt")
                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
                c = cert.diagnostics
                blo, bhi = BECKER_T2[(d, S0)]
                lo, hi = c["ci95"]
                overlap = lo <= bhi and blo <= hi
                print(f"  {variant:14s} L0={c['lower']:9.4f} ({c['lower_stderr']:.4f})  "
                      f"gap={c['gap']:.4f}  CI=[{lo:.3f}, {hi:.3f}]  "
                      f"{'overlaps Becker' if overlap else '*** NO OVERLAP with Becker ***'}"
                      f"  fit {fit_s:.0f}s cert {cert.runtime:.0f}s", flush=True)
                row = {"phase": phase, "variant": variant, "d": d, "S0": S0, "seed": seed,
                       "budget": budget, "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                       "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                       "U0": c["upper"], "gap": c["gap"], "ci_lo": lo, "ci_hi": hi,
                       "becker_lo": blo, "becker_hi": bhi, "overlaps_becker": overlap,
                       "fit_s": round(fit_s, 1), "cert_s": round(cert.runtime, 1),
                       "timestamp": datetime.now().isoformat(timespec="seconds")}
                append_row(row)
                rows[(variant, d, S0)] = {k: str(v) for k, v in row.items()}
            compare(rows, d, S0)
        if phase == "validation" and not SMOKE:
            bad = [k for k, r in rows.items() if r["phase"] == "validation"
                   and k[0] in ("lsm", "nn_lsm", "becker") and r["overlaps_becker"] != "True"]
            if bad:
                print(f"\n*** VALIDATION FAILED for {bad}: main phase not run. ***")
                return
            print("\nvalidation passed: every sweep variant overlaps Becker et al. at d = 2, 3")

    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
