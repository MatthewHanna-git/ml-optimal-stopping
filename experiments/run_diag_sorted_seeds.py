"""Diagnostic: the seed spread of NN-LSM and DOS with sorted inputs.

Each cell of run_diag_sorted_inputs.py is a single seed. This retrains the
sorted NN-LSM and sorted DOS at d in {10, 50}, S0 = 100, from the five
reseeds of the seed arm (run_diag_seed_arm.py), so each sorted draw j shares
its seed, and so its training and certification streams, with the unsorted
draw j and with LSM's draw j. With the sweep-seed cell from
diag_sorted_inputs.csv, each (method, d) has six draws, paired in order with
the six of the seed arm exactly as in the thesis's seed table.

Two comparisons are read, each standing where its sign holds in all six
repeats (or none), as for the other contrasts of the seed table:

    sorted less unsorted   the effect of sorting, same method
    sorted less LSM        whether the sorted network leads LSM

Settings, budgets and certification are the sweep's, as in
run_diag_sorted_inputs.py, whose sorted networks this imports.

Writes results/diag_sorted_seeds.csv (resumable) and the networks to
results/nets_diag/. About 35 minutes on an RTX 3080.

Usage:  python experiments/run_diag_sorted_seeds.py
        python experiments/run_diag_sorted_seeds.py --smoke
"""

import csv
import dataclasses
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch

import run_sweep as sweep
import run_diag_sorted_inputs as srt     # patches nn_lsm / becker to sorted inputs
from src import payoffs
from src.methods import nn_lsm, becker
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_sorted_seeds.csv" if SMOKE else "diag_sorted_seeds.csv")
NETS_DIR = RESULTS / ("nets_diag_smoke" if SMOKE else "nets_diag")
SEED_ARM_CSV = RESULTS / "diag_seed_arm.csv"
SORTED_CSV = RESULTS / "diag_sorted_inputs.csv"

METHODS = ["nn_lsm", "becker"]
DIMS = [10, 50]
S0 = 100.0
N_SEEDS = 5
# The seed arm's seeds: k over (method in its order, d, j), s = 3e6 + 1e4 k
ARM_METHODS = ["lsm", "nn_lsm", "becker", "hure"]
SEED_BASE, SEED_STEP = 3_000_000, 10_000

FIELDS = ["method", "d", "S0", "j", "seed", "budget",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "unsorted_L0", "lsm_L0", "minus_unsorted", "minus_lsm",
          "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def arm_seed(method, d, j):
    k = (ARM_METHODS.index(method) * 2 + [10, 50].index(d)) * N_SEEDS + j
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
    return {(r["method"], int(r["d"]), r["j"]) for r in read_csv(CSV_PATH)}


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
    sorted_cells = {(r["method"], int(r["d"])): r for r in read_csv(SORTED_CSV)
                    if float(r["S0"]) == S0}
    print("\nSix draws per (method, d): the sweep's seed, then j = 0..4. "
          "Range of each difference and repeats with it positive.")
    for m in METHODS:
        for d in DIMS:
            draws = []
            sc = sorted_cells.get((m, d))
            base = arm.get((m, d, "sweep")), arm.get(("lsm", d, "sweep"))
            if sc and all(base):
                draws.append((float(sc["L0"]), float(base[0]["L0"]), float(base[1]["L0"])))
            for r in rows:
                if r["method"] == m and int(r["d"]) == d:
                    draws.append((float(r["L0"]), float(r["unsorted_L0"]), float(r["lsm_L0"])))
            if not draws:
                continue
            du = [a - b for a, b, _ in draws]
            dl = [a - c for a, _, c in draws]
            s = [a for a, _, _ in draws]
            print(f"  {m:7s} d={d:<3d} n={len(draws)}  sorted L0 {min(s):.3f} to {max(s):.3f}  |  "
                  f"less unsorted {min(du):+.3f} to {max(du):+.3f} ({sum(x > 0 for x in du)}/{len(du)})  |  "
                  f"less LSM {min(dl):+.3f} to {max(dl):+.3f} ({sum(x > 0 for x in dl)}/{len(dl)})")


def main():
    sweep.env_banner()
    print(f"diag csv: {CSV_PATH}\ndiag nets: {NETS_DIR}")
    arm = arm_rows()
    if not SMOKE:
        for m in METHODS + ["lsm"]:
            for d in DIMS:
                for j in range(N_SEEDS):
                    r = arm.get((m, d, j))
                    assert r is not None, f"seed arm row missing: {m} d={d} j={j}"
                    assert int(r["seed"]) == arm_seed(m, d, j), f"seed mismatch: {m} d={d} j={j}"
        print("seed check: all 30 seed-arm rows found, seeds match its formula")
    done = load_done()
    if done:
        print(f"resuming: {len(done)} runs already recorded, will be skipped")
    t_all = time.perf_counter()

    for m in METHODS:
        for d in DIMS:
            for j in range(N_SEEDS):
                if (m, d, str(j)) in done:
                    print(f"skip (done): {m} d={d} j={j}")
                    continue
                seed = arm_seed(m, d, j)
                spd = sweep.SPD[m][d]
                params = sweep.make_params(d, seed, dict(n_iters=spd * 8,
                                                         batch_size=sweep.BATCH_ML,
                                                         lr=sweep.LR_ML))
                torch.manual_seed(seed)
                sweep._sync(); t0 = time.perf_counter()
                if m == "nn_lsm":
                    policy, _, _ = nn_lsm.fit(S0, params, device=DEVICE, n_train=sweep.N_TRAIN)
                else:
                    policy, _ = becker.fit(S0, params, device=DEVICE)
                sweep._sync(); fit_s = time.perf_counter() - t0

                NETS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"method": m, "d": d, "S0": S0, "seed": seed, "j": j,
                            "inputs": "sorted descending",
                            "params": dataclasses.asdict(params),
                            "nets": sweep._state(policy.nets)},
                           NETS_DIR / f"{m}_sorted_seed{j}_d{d}_S{S0:.0f}.pt")
                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
                c = cert.diagnostics

                u = arm.get((m, d, j)); l = arm.get(("lsm", d, j))
                u_L0 = float(u["L0"]) if u and not SMOKE else float("nan")
                l_L0 = float(l["L0"]) if l and not SMOKE else float("nan")
                print(f"{m:7s} d={d:<3d} j={j} seed={seed}: L0={c['lower']:.4f} ({c['lower_stderr']:.4f})"
                      f"  gap={c['gap']:.4f}  | unsorted {u_L0:.4f} ({c['lower'] - u_L0:+.4f})"
                      f"  LSM {l_L0:.4f} ({c['lower'] - l_L0:+.4f})"
                      f"\n{'':8s}fit {fit_s:.0f}s  cert {cert.runtime:.0f}s", flush=True)
                append_row({
                    "method": m, "d": d, "S0": S0, "j": j, "seed": seed,
                    "budget": f"{spd}/date",
                    "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                    "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                    "U0": c["upper"], "gap": c["gap"],
                    "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1],
                    "unsorted_L0": u_L0, "lsm_L0": l_L0,
                    "minus_unsorted": c["lower"] - u_L0, "minus_lsm": c["lower"] - l_L0,
                    "fit_s": round(fit_s, 1), "cert_s": round(cert.runtime, 1),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })

    if not SMOKE:
        summarise(arm)
    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
