"""Diagnostic: the seed spread of LSM on the generic basis.

Each cell of run_diag_generic_lsm.py is a single seed, the sweep's. This
refits LSM on the generic basis with the payoff (poly2g) at d in {10, 50}, S0 = 100, from the five
reseeds of the seed arm (run_diag_seed_arm.py). Each draw j uses LSM's own
seed-arm seed for j, so it shares its training paths and certification
streams with the seed arm's LSM draw j and differs from it only in the
basis. With the sweep-seed cell from diag_generic_lsm.csv, each d has six
draws, paired in order exactly as in the thesis's seed table and as in
run_diag_sorted_seeds.py.

LSM less generic, the cost of taking the structure from LSM, stands where
its sign holds in all six repeats (or none).

Basis, regression and certification are those of run_diag_generic_lsm.py,
whose GenericBasis and fit this imports.

Writes results/diag_generic_seeds.csv (resumable). About half an hour on
an RTX 3080, nearly all of it certification.

Usage:  python experiments/run_diag_generic_seeds.py
        python experiments/run_diag_generic_seeds.py --smoke
"""

import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import run_sweep as sweep
import run_diag_generic_lsm as gen       # GenericBasis and fit, the sweep's LSM recursion
from src import payoffs
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_generic_seeds.csv" if SMOKE else "diag_generic_seeds.csv")
SEED_ARM_CSV = RESULTS / "diag_seed_arm.csv"
GENERIC_CSV = RESULTS / "diag_generic_lsm.csv"

BASIS = ("poly2g", 2, True)
DIMS = [10] if SMOKE else [10, 50]
S0 = 100.0
N_SEEDS = 1 if SMOKE else 5
# The seed arm's seeds: k over (method in its order, d, j), s = 3e6 + 1e4 k
ARM_METHODS = ["lsm", "nn_lsm", "becker", "hure"]
SEED_BASE, SEED_STEP = 3_000_000, 10_000

FIELDS = ["basis", "d", "S0", "j", "seed", "n_basis", "n_train",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "in_sample", "skipped_dates",
          "lsm_L0", "lsm_L0_stderr", "lsm_minus_generic", "z",
          "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def arm_seed(method, d, j):
    k = (ARM_METHODS.index(method) * 2 + [10, 50].index(d)) * 5 + j
    return SEED_BASE + SEED_STEP * k


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def arm_rows():
    """{(method, d, j or 'sweep'): row} from the seed arm."""
    out = {}
    for r in read_csv(SEED_ARM_CSV):
        j = "sweep" if r["source"] == "sweep_results.csv" else int(r["j"])
        out[(r["method"], int(r["d"]), j)] = r
    return out


def load_done():
    return {(int(r["d"]), r["j"]) for r in read_csv(CSV_PATH)}


def append_row(row):
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def summarise(arm):
    rows = read_csv(CSV_PATH)
    sweep_cells = {int(r["d"]): r for r in read_csv(GENERIC_CSV)
                   if r["basis"] == BASIS[0] and float(r["S0"]) == S0}
    print("\nSix draws per d: the sweep's seed, then j = 0..4. "
          "Range of LSM less generic and repeats with it positive.")
    for d in DIMS:
        diffs = []
        sc, base = sweep_cells.get(d), arm.get(("lsm", d, "sweep"))
        if sc and base:
            diffs.append(float(base["L0"]) - float(sc["L0"]))
        diffs += [float(r["lsm_minus_generic"]) for r in rows if int(r["d"]) == d]
        if diffs:
            print(f"  d={d:<3d} n={len(diffs)}  LSM less generic {min(diffs):+.3f} to "
                  f"{max(diffs):+.3f}  ({sum(x > 0 for x in diffs)}/{len(diffs)} positive)")


def main():
    sweep.env_banner()
    print(f"diag csv: {CSV_PATH}")
    arm = arm_rows()
    if not SMOKE:
        for d in DIMS:
            for j in range(N_SEEDS):
                r = arm.get(("lsm", d, j))
                assert r is not None, f"seed arm row missing: lsm d={d} j={j}"
                assert int(r["seed"]) == arm_seed("lsm", d, j), f"seed mismatch: lsm d={d} j={j}"
        print("seed check: all 10 LSM seed-arm rows found, seeds match its formula")
    done = load_done()
    if done:
        print(f"resuming: {len(done)} runs already recorded, will be skipped")
    t_all = time.perf_counter()
    name, degree, with_g = BASIS

    for d in DIMS:
        for j in range(N_SEEDS):
            if (d, str(j)) in done:
                print(f"skip (done): d={d} j={j}")
                continue
            seed = arm_seed("lsm", d, j)                    # LSM's own seed-arm draw j
            params = sweep.make_params(d, seed, {})
            basis = gen.GenericBasis(d, degree, with_g, params.K_strike)
            sweep._sync(); t0 = time.perf_counter()
            policy, v_in = gen.fit(S0, params, basis, sweep.N_TRAIN)
            sweep._sync(); fit_s = time.perf_counter() - t0
            cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
            c = cert.diagnostics

            l = None if SMOKE else arm.get(("lsm", d, j))
            l_L0 = float(l["L0"]) if l else float("nan")
            l_se = float(l["L0_stderr"]) if l else float("nan")
            diff = l_L0 - c["lower"]
            se = math.sqrt(c["lower_stderr"] ** 2 + l_se ** 2)
            z = diff / se if se > 0 else float("nan")

            print(f"{name} d={d:<3d} j={j} seed={seed} B={basis.size}: "
                  f"L0={c['lower']:.4f} ({c['lower_stderr']:.4f})  gap={c['gap']:.4f}"
                  f"  | LSM {l_L0:.4f}, LSM less generic {diff:+.4f} (z={z:+.1f})"
                  f"{'  skipped dates ' + str(policy.skipped_dates) if policy.skipped_dates else ''}"
                  f"\n{'':7s}fit {fit_s:.0f}s  cert {cert.runtime:.0f}s", flush=True)
            append_row({
                "basis": name, "d": d, "S0": S0, "j": j, "seed": seed,
                "n_basis": basis.size, "n_train": sweep.N_TRAIN,
                "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                "U0": c["upper"], "gap": c["gap"],
                "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1],
                "in_sample": v_in, "skipped_dates": " ".join(map(str, policy.skipped_dates)),
                "lsm_L0": l_L0, "lsm_L0_stderr": l_se,
                "lsm_minus_generic": diff, "z": z,
                "fit_s": round(fit_s, 1), "cert_s": round(cert.runtime, 1),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

    if not SMOKE:
        summarise(arm)
    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
