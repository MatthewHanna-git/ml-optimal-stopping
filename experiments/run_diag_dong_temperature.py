"""Diagnostic: Dong and Chee trained at a constant temperature.

Dong (2024, Table 4.2a) prices the sweep's contract and trains at a
constant lambda = 1e-2, where the sweep anneals the temperature from 0.1.
This reruns the sweep's Dong cells at d in {5, 10, 50}, S0 = 100, and
Chee's at d = 5, S0 = 100, with the temperature held at 1e-2 from the
first iteration (Params.lambda_start = lambda_). Everything else is the
sweep cell's: seed, budget, learning rate, network, certification streams
and sample sizes.

Each cell's certified L0 is compared with the sweep's and with Dong's
published value. A rise is detectable if it exceeds DETECT, and explains the
shortfall if it closes at least half the gap to the published value.

Writes results/diag_dong_temperature.csv (appended per cell, resumable)
and the networks to results/nets_diag/.

Usage:  python experiments/run_diag_dong_temperature.py
        python experiments/run_diag_dong_temperature.py --smoke
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

import run_sweep as sweep                      # seeds, params, settings
from src import payoffs
from src.methods import dong, chee
from src.methods.rl_policies import GreedyValuePolicy
from src.methods.rl_sweep_eval import randomised_value
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_dong_temperature.csv" if SMOKE
                      else "diag_dong_temperature.csv")
NETS_DIR = RESULTS / ("nets_diag_smoke" if SMOKE else "nets_diag")
SWEEP_CSV = RESULTS / "sweep_results.csv"

CELLS = [("dong", 5, 100.0), ("dong", 10, 100.0), ("dong", 50, 100.0),
         ("chee", 5, 100.0)]
if SMOKE:
    CELLS = [("dong", 2, 100.0), ("chee", 2, 100.0)]

DETECT = 0.06            # above the noise of a single training seed

# Dong (2024), Table 4.2a, symmetric case: test-set payoff on 2^18 paths.
DONG_T42 = {(2, 90.0): 8.067, (2, 100.0): 13.871, (2, 110.0): 21.352,
            (5, 90.0): 16.646, (5, 100.0): 26.161, (5, 110.0): 36.720,
            (10, 90.0): 26.240, (10, 100.0): 38.255, (10, 110.0): 50.808,
            (50, 90.0): 53.983, (50, 100.0): 69.528, (50, 110.0): 85.185}
DONG_TEST_PATHS = 2 ** 18

FIELDS = ["method", "d", "S0", "schedule", "settings", "budget", "seed",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap",
          "ci_lo", "ci_hi", "randomized", "randomized_stderr", "own_greedy",
          "sweep_L0", "sweep_L0_stderr", "sweep_gap", "dL0", "dL0_se",
          "dong_t42", "dong_se_recovered", "shortfall_to_dong",
          "detectable", "explains", "fit_s", "eval_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def load_sweep_rows():
    rows = {}
    if not SWEEP_CSV.exists():
        return rows
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
    print("RULE: detectable iff dL0 > %.2f; explains iff dL0 >= 0.5 * (Dong T4.2a - sweep L0)"
          % DETECT)
    srows = load_sweep_rows()
    done = load_done()
    if done:
        print(f"resuming: {len(done)} cells already recorded, will be skipped")
    t_all = time.perf_counter()

    for method, d, S0 in CELLS:
        key = (method, d, S0)
        if key in done:
            print(f"skip (done): {method} d={d} S0={S0:.0f}")
            continue
        mod = {"dong": dong, "chee": chee}[method]
        seed = sweep.cell_seed(method, d, S0)          # the sweep's own seed
        setting = dict(sweep.RL_SETTINGS[method])
        lam = setting["lambda_"]
        params = sweep.make_params(d, seed, {**setting, **sweep.RL_BUDGET,
                                             "lambda_start": lam})
        assert params.lambda_start == params.lambda_
        torch.manual_seed(seed)
        sweep._sync(); t0 = time.perf_counter()
        nets, diag = mod.fit(S0, params, payoff_fn=payoffs.max_call, device=DEVICE)
        sweep._sync(); fit_s = time.perf_counter() - t0
        assert diag["anneal"].startswith(f"{lam:g} -> {lam:g}"), diag["anneal"]

        n_rand = min(sweep.N_EVAL, sweep.RL_EVAL_CAP) if d >= 50 else sweep.N_EVAL
        ev = randomised_value(method, nets, S0, params, payoffs.max_call, DEVICE,
                              seed=seed + sweep.OFF_RAND, n_paths=n_rand,
                              threshold=sweep.E_INV)
        policy = GreedyValuePolicy(params, payoffs.max_call, nets)

        NETS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"method": method, "d": d, "S0": S0, "seed": seed,
                    "schedule": f"constant lambda={lam:g}",
                    "params": dataclasses.asdict(params),
                    "nets": sweep._state(nets)},
                   NETS_DIR / f"{method}_constlam_d{d}_S{S0:.0f}.pt")

        cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
        c = cert.diagnostics
        lo, hi = c["ci95"]

        sr = srows.get(key)
        sweep_L0 = float(sr["L0"]) if sr else float("nan")
        sweep_se = float(sr["L0_stderr"]) if sr else float("nan")
        sweep_gap = float(sr["gap"]) if sr else float("nan")
        dL0 = c["lower"] - sweep_L0
        dL0_se = math.sqrt(c["lower_stderr"] ** 2 + sweep_se ** 2)
        anchor = DONG_T42.get((d, S0)) if method == "dong" else None
        dong_se = (sweep_se * math.sqrt(sweep.N_EVAL / DONG_TEST_PATHS)
                   if anchor is not None else float("nan"))
        shortfall = (anchor - sweep_L0) if anchor is not None else float("nan")
        detectable = dL0 > DETECT
        explains = (anchor is not None) and (dL0 >= 0.5 * shortfall)

        print(f"{method:5s} d={d:<3d} S0={S0:.0f} seed={seed}  CONSTANT lambda={lam:g}\n"
              f"      L0={c['lower']:.4f} ({c['lower_stderr']:.4f})  U0={c['upper']:.4f}  "
              f"gap={c['gap']:.4f}  CI=[{lo:.4f}, {hi:.4f}]  rand={ev['value']:.4f}\n"
              f"      sweep (annealed): L0={sweep_L0:.4f} ({sweep_se:.4f})  gap={sweep_gap:.4f}\n"
              f"      dL0={dL0:+.4f} (combined s.e. {dL0_se:.4f})  "
              f"detectable(>{DETECT}): {detectable}")
        if anchor is not None:
            print(f"      Dong T4.2a: {anchor:.3f} (s.e. ~{dong_se:.3f} on 2^18 paths)  "
                  f"shortfall sweep->Dong {shortfall:+.3f}  explains(>=half): {explains}")
        print(f"      fit {fit_s:.0f}s  eval {ev['eval_s']:.0f}s  cert {cert.runtime:.0f}s")

        append_row({
            "method": method, "d": d, "S0": S0,
            "schedule": f"constant lambda={lam:g}",
            "settings": ", ".join(f"{k}={v:g}" for k, v in setting.items()),
            "budget": f"{sweep.RL_BUDGET['n_iters']}/{sweep.RL_BUDGET['batch_size']}",
            "seed": seed,
            "L0": c["lower"], "L0_stderr": c["lower_stderr"],
            "delta": c["delta"], "delta_stderr": c["delta_stderr"],
            "U0": c["upper"], "gap": c["gap"], "ci_lo": lo, "ci_hi": hi,
            "randomized": ev["value"], "randomized_stderr": ev["mc_stderr"],
            "own_greedy": ev["greedy_value"],
            "sweep_L0": sweep_L0, "sweep_L0_stderr": sweep_se, "sweep_gap": sweep_gap,
            "dL0": dL0, "dL0_se": dL0_se,
            "dong_t42": "" if anchor is None else anchor,
            "dong_se_recovered": "" if anchor is None else dong_se,
            "shortfall_to_dong": "" if anchor is None else shortfall,
            "detectable": detectable, "explains": explains,
            "fit_s": round(fit_s, 1), "eval_s": round(ev["eval_s"], 1),
            "cert_s": round(cert.runtime, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
