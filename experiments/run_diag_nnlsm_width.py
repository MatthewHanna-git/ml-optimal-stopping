"""Diagnostic: NN-LSM at d = 50 by network width and training paths.

Lapeyre and Lelong (2021, Tables 10-11) price this contract at d = 50 with
128 to 512 neurons, where the sweep uses two layers of 50. This retrains the
sweep's cell (NN-LSM, d = 50, S0 = 100) on a 2 x 2 grid,

    width in {50, 256}  x  training paths in {2^18, 2^20},

keeping the sweep's two-layer class, budget (3000 steps per date on batches
of 8192), learning rate, seed, certification streams and sample sizes. The
(50, 2^18) cell is the sweep's and is read from sweep_results.csv unless
RERUN_BASE is set. The paper's in-sample estimate is recorded beside each
certified L0.

A rise in L0 over the sweep's cell is detectable if it exceeds DETECT, and
material if it reaches MATERIAL, half the shortfall to LSM.

Writes results/diag_nnlsm_width.csv (appended per cell, resumable) and the
networks to results/nets_diag/. At 2^20 paths the path tensor is about
2.1 GB.

Usage:  python experiments/run_diag_nnlsm_width.py
        python experiments/run_diag_nnlsm_width.py --smoke
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
from src.methods import nn_lsm
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_nnlsm_width.csv" if SMOKE else "diag_nnlsm_width.csv")
NETS_DIR = RESULTS / ("nets_diag_smoke" if SMOKE else "nets_diag")
SWEEP_CSV = RESULTS / "sweep_results.csv"

D, S0 = 50, 100.0
WIDTHS = [50, 256]
N_TRAINS = [2 ** 18, 2 ** 20]
RERUN_BASE = False           # True: also retrain the sweep's (50, 2^18) cell
DETECT, MATERIAL = 0.06, 0.22
LAPEYRE = "Lapeyre-Lelong T10 (M=1e5) / T11 (M=1e6), one hidden layer of 128-512: 68.99-69.51 / 68.82-69.19 in-sample"

if SMOKE:
    D = 5
    WIDTHS = [50, 64]
    N_TRAINS = [2 ** 12, 2 ** 13]

FIELDS = ["method", "d", "S0", "width", "n_train", "budget", "seed", "source",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "in_sample", "in_sample_stderr",
          "base_L0", "base_L0_stderr", "dL0", "dL0_se", "detectable", "material",
          "lsm_L0", "shortfall_to_lsm",
          "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def load_sweep_rows():
    rows = {}
    if SWEEP_CSV.exists():
        with open(SWEEP_CSV, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["method"], int(r["d"]), float(r["S0"]))] = r
    return rows


def load_done():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as f:
        return {(int(r["width"]), int(r["n_train"])) for r in csv.DictReader(f)}


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
    print(f"RULE: on L0 against the sweep's (50, 2^{N_TRAINS[0].bit_length() - 1}) cell: "
          f"detectable iff rise > {DETECT}; material iff rise >= {MATERIAL}")
    srows = load_sweep_rows()
    base = srows.get(("nn_lsm", D, S0))
    lsm_row = srows.get(("lsm", D, S0))
    base_L0 = float(base["L0"]) if base else float("nan")
    base_se = float(base["L0_stderr"]) if base else float("nan")
    lsm_L0 = float(lsm_row["L0"]) if lsm_row else float("nan")
    if base:
        print(f"sweep cell nn_lsm d={D} S0={S0:.0f}: L0={base_L0:.4f} ({base_se:.4f}) "
              f"gap={float(base['gap']):.4f}; LSM L0={lsm_L0:.4f}; {LAPEYRE}")
    done = load_done()
    if done:
        print(f"resuming: {len(done)} cells already recorded, will be skipped")
    t_all = time.perf_counter()

    seed = sweep.cell_seed("nn_lsm", D, S0)        # the sweep's own seed
    spd = sweep.SPD["nn_lsm"][D]

    for width in WIDTHS:
        for n_train in N_TRAINS:
            key = (width, n_train)
            if key in done:
                print(f"skip (done): width={width} n_train=2^{n_train.bit_length() - 1}")
                continue
            is_base = (width == WIDTHS[0] and n_train == N_TRAINS[0])
            if is_base and not RERUN_BASE and base is not None and not SMOKE:
                c = {"lower": base_L0, "lower_stderr": base_se,
                     "delta": float(base["delta"]), "delta_stderr": float(base["delta_stderr"]),
                     "upper": float(base["U0"]), "gap": float(base["gap"]),
                     "ci95": (float(base["ci_lo"]), float(base["ci_hi"]))}
                v_in, se_in, fit_s, cert_s, source = "", "", float(base["fit_s"]), \
                    float(base["cert_s"]), "sweep_results.csv"
                print(f"width={width:<4d} n_train=2^{n_train.bit_length() - 1}: the sweep's row, "
                      f"L0={c['lower']:.4f} ({c['lower_stderr']:.4f}) gap={c['gap']:.4f}")
            else:
                params = sweep.make_params(D, seed, dict(n_iters=spd * 8,
                                                         batch_size=sweep.BATCH_ML,
                                                         lr=sweep.LR_ML,
                                                         arch=(width, width)))
                torch.manual_seed(seed)
                sweep._sync(); t0 = time.perf_counter()
                policy, v_in, se_in = nn_lsm.fit(S0, params, device=DEVICE, n_train=n_train)
                sweep._sync(); fit_s = time.perf_counter() - t0
                NETS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"method": "nn_lsm", "d": D, "S0": S0, "seed": seed,
                            "width": width, "n_train": n_train,
                            "params": dataclasses.asdict(params),
                            "nets": sweep._state(policy.nets)},
                           NETS_DIR / f"nn_lsm_w{width}_n{n_train.bit_length() - 1}_d{D}_S{S0:.0f}.pt")
                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
                c, cert_s, source = cert.diagnostics, cert.runtime, "this run"
                print(f"width={width:<4d} n_train=2^{n_train.bit_length() - 1}: "
                      f"L0={c['lower']:.4f} ({c['lower_stderr']:.4f})  U0={c['upper']:.4f}  "
                      f"gap={c['gap']:.4f}  CI=[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]  "
                      f"in-sample={v_in:.4f} ({se_in:.4f})\n"
                      f"      fit {fit_s:.0f}s  cert {cert_s:.0f}s")

            dL0 = c["lower"] - base_L0
            dL0_se = math.sqrt(c["lower_stderr"] ** 2 + base_se ** 2)
            detectable, material = dL0 > DETECT, dL0 >= MATERIAL
            if not is_base:
                print(f"      dL0 vs sweep cell {dL0:+.4f} (combined s.e. {dL0_se:.4f})  "
                      f"detectable: {detectable}  material: {material}")
            append_row({
                "method": "nn_lsm", "d": D, "S0": S0, "width": width, "n_train": n_train,
                "budget": f"{spd}/date", "seed": seed, "source": source,
                "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                "U0": c["upper"], "gap": c["gap"],
                "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1],
                "in_sample": v_in, "in_sample_stderr": se_in,
                "base_L0": base_L0, "base_L0_stderr": base_se,
                "dL0": dL0, "dL0_se": dL0_se,
                "detectable": detectable, "material": material,
                "lsm_L0": lsm_L0, "shortfall_to_lsm": lsm_L0 - c["lower"],
                "fit_s": round(fit_s, 1), "cert_s": round(cert_s, 1),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
