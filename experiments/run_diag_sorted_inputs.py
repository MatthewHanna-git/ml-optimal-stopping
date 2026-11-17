"""Diagnostic: NN-LSM and DOS with the asset prices sorted before the network.

The contract is symmetric, so the continuation value and the exercise
decision depend on the set of prices, not on which asset holds which. LSM's
basis encodes this: it is built on the prices in descending order. The
networks of the sweep receive the prices in a fixed asset order and must
learn the symmetry from data. This run hands it to them.

Each cell is the sweep's cell with one change: the prices enter the network
in descending order. The payoff input, max of the prices, is unchanged by
sorting. Everything else is the sweep's: the 2 x 50 trunk, budgets, learning
rate, training paths, seed (so the same initial weights and training draws)
and certification streams (so each sorted cell is valued on the same paths
as its unsorted twin).

Recorded against the sweep's cell on the same paths:

    dL0     = L0(sorted) - L0(sweep)
    closed  = dL0 / (L0(LSM) - L0(sweep)), the share of the shortfall to LSM

A change is read only where |dL0| exceeds 2 combined standard errors
(conservative, since both share paths).

Writes results/diag_sorted_inputs.csv (appended per cell, resumable) and the
networks to results/nets_diag/ (loading them requires the sorted wrappers
below).

Usage:  python experiments/run_diag_sorted_inputs.py
        python experiments/run_diag_sorted_inputs.py --smoke
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
from src import payoffs
from src.methods import nn_lsm, becker
from src.methods.common import ValueNet, StopNet
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_sorted_inputs.csv" if SMOKE else "diag_sorted_inputs.csv")
NETS_DIR = RESULTS / ("nets_diag_smoke" if SMOKE else "nets_diag")
SWEEP_CSV = RESULTS / "sweep_results.csv"

METHODS = ["nn_lsm", "becker"]
DIMS = list(sweep.DIMS)                 # [2, 3, 5, 10, 20, 50]; smoke: [2, 5]
S0_LIST = sweep.S0_LIST
READ_Z = 2.0

FIELDS = ["method", "d", "S0", "inputs", "seed", "budget",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "sweep_L0", "sweep_L0_stderr", "sweep_gap", "dL0", "dL0_se", "z", "read",
          "lsm_L0", "shortfall_sweep", "shortfall_sorted", "closed",
          "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def _desc(S):
    return torch.sort(S, dim=-1, descending=True).values


class SortedValueNet(ValueNet):
    """ValueNet fed the prices in descending order."""

    def forward(self, S):
        return super().forward(_desc(S))


class SortedStopNet(StopNet):
    """StopNet fed the prices in descending order."""

    def forward(self, S):
        return super().forward(_desc(S))


# fit() builds its networks by these module-level names; nothing else changes.
nn_lsm.ValueNet = SortedValueNet
becker.StopNet = SortedStopNet


def load_sweep_rows():
    rows = {}
    with open(SWEEP_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["method"], int(r["d"]), float(r["S0"]))] = r
    return rows


def load_done():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as f:
        return {(r["method"], int(r["d"]), float(r["S0"])) for r in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def main():
    sweep.env_banner()
    print(f"diag csv: {CSV_PATH}\ndiag nets: {NETS_DIR}")
    print(f"RULE: read dL0 only where |dL0| > {READ_Z:g} x combined s.e. "
          f"(conservative: sorted and sweep cells share paths)")
    srows = load_sweep_rows()
    done = load_done()
    if done:
        print(f"resuming: {len(done)} cells already recorded, will be skipped")
    t_all = time.perf_counter()

    for method in METHODS:
        for d in DIMS:
            for S0 in S0_LIST:
                key = (method, d, S0)
                if key in done:
                    print(f"skip (done): {method} d={d} S0={S0:.0f}")
                    continue
                seed = sweep.cell_seed(method, d, S0)     # the sweep cell's own seed
                spd = sweep.SPD[method][d]
                params = sweep.make_params(d, seed, dict(n_iters=spd * 8,
                                                         batch_size=sweep.BATCH_ML,
                                                         lr=sweep.LR_ML))
                torch.manual_seed(seed)                     # same initial weights
                sweep._sync(); t0 = time.perf_counter()
                if method == "nn_lsm":
                    policy, _, _ = nn_lsm.fit(S0, params, device=DEVICE,
                                              n_train=sweep.N_TRAIN)
                else:
                    policy, _ = becker.fit(S0, params, device=DEVICE)
                sweep._sync(); fit_s = time.perf_counter() - t0

                NETS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"method": method, "d": d, "S0": S0, "seed": seed,
                            "inputs": "sorted descending",
                            "params": dataclasses.asdict(params),
                            "nets": sweep._state(policy.nets)},
                           NETS_DIR / f"{method}_sorted_d{d}_S{S0:.0f}.pt")

                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call,
                               device=DEVICE)
                c = cert.diagnostics

                base = srows.get(key)
                lsm_row = srows.get(("lsm", d, S0))
                if base is None or SMOKE:
                    b_L0 = b_se = b_gap = lsm_L0 = float("nan")
                else:
                    b_L0, b_se = float(base["L0"]), float(base["L0_stderr"])
                    b_gap = float(base["gap"])
                    lsm_L0 = float(lsm_row["L0"]) if lsm_row else float("nan")
                dL0 = c["lower"] - b_L0
                dL0_se = math.sqrt(c["lower_stderr"] ** 2 + b_se ** 2)
                z = dL0 / dL0_se if dL0_se > 0 else float("nan")
                read = abs(z) > READ_Z
                short_b, short_s = lsm_L0 - b_L0, lsm_L0 - c["lower"]
                closed = dL0 / short_b if short_b > 0 else float("nan")

                print(f"{method:7s} d={d:<3d} S0={S0:<4.0f} sorted: "
                      f"L0={c['lower']:9.4f} ({c['lower_stderr']:.4f})  gap={c['gap']:.4f}  "
                      f"| sweep L0={b_L0:9.4f} gap={b_gap:.4f}  dL0={dL0:+.4f} (z={z:+.1f}"
                      f"{', read' if read else ''})  "
                      f"| LSM {lsm_L0:.4f}: shortfall {short_b:.4f} -> {short_s:.4f}"
                      f" (closed {closed:+.0%})\n"
                      f"{'':8s}fit {fit_s:.0f}s  cert {cert.runtime:.0f}s", flush=True)

                append_row({
                    "method": method, "d": d, "S0": S0, "inputs": "sorted",
                    "seed": seed, "budget": f"{spd}/date",
                    "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                    "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                    "U0": c["upper"], "gap": c["gap"],
                    "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1],
                    "sweep_L0": b_L0, "sweep_L0_stderr": b_se, "sweep_gap": b_gap,
                    "dL0": dL0, "dL0_se": dL0_se, "z": z, "read": read,
                    "lsm_L0": lsm_L0, "shortfall_sweep": short_b,
                    "shortfall_sorted": short_s, "closed": closed,
                    "fit_s": round(fit_s, 1), "cert_s": round(cert.runtime, 1),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })

    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
