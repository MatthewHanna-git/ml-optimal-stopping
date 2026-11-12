"""Diagnostic: Dai, Chee and Dianetti at double the iterations, at d = 20.

The counterpart of run_diag_dong_budget.py for the other three RL methods,
at the cell of the ML depth calibration (d = 20, S0 = 100). Each method is
retrained with one change: 12000 iterations, the batch held at 2048.
Everything else is the sweep cell's: seed, settings, Dianetti's learning
rate, annealing, network, randomised evaluation, certified rule (Dianetti
at e^{-1}), certification streams and sample sizes. Dong at d = 20 can be
added through CELLS.

The schedules are proportional to n_iters, as in run_diag_dong_budget.py,
and Dianetti's softplus width follows the temperature.

The calibration rule of the other methods is applied as it stands: the
doubled budget would be adopted if gap(2B) <= max{0.15, gap(B)/2}, with
gap(B) the sweep's. The rise dL0 = L0(2B) - L0(B) is reported beside it,
against DETECT. One training seed per cell, paired with the sweep's.

Writes results/diag_rl_budget.csv (appended per cell, resumable) and the
networks to results/nets_diag/.

Usage:  python experiments/run_diag_rl_budget.py
        python experiments/run_diag_rl_budget.py --smoke
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
from src.methods import dong, dai, chee, dianetti
from src.methods.rl_policies import GreedyValuePolicy, FuelLevelPolicy
from src.methods.rl_sweep_eval import randomised_value
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_rl_budget.csv" if SMOKE else "diag_rl_budget.csv")
NETS_DIR = RESULTS / ("nets_diag_smoke" if SMOKE else "nets_diag")
SWEEP_CSV = RESULTS / "sweep_results.csv"

CELLS = [("dai", 20, 100.0), ("chee", 20, 100.0), ("dianetti", 20, 100.0)]
# CELLS.append(("dong", 20, 100.0))      # optional: the same cell for Dong
if SMOKE:
    CELLS = [("dai", 2, 100.0), ("dianetti", 2, 100.0)]

MODS = {"dong": dong, "dai": dai, "chee": chee, "dianetti": dianetti}
GAP_FLOOR = 0.15         # the calibration rule's absolute threshold
DETECT = 0.06            # reported alongside, as in the other diagnostics
RULE = ("RULE: adopt 2B iff gap(2B) <= max{%.2f, 0.5 * gap(B)}, to the letter, no overrides; "
        "detectable (reported only) iff dL0 = L0(2B) - L0(B) > %.2f" % (GAP_FLOOR, DETECT))

FIELDS = ["method", "d", "S0", "budget_arm", "realised_schedule", "settings", "budget",
          "seed", "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap",
          "ci_lo", "ci_hi", "randomized", "randomized_stderr", "own_greedy",
          "sweep_L0", "sweep_L0_stderr", "sweep_gap", "dL0", "dL0_se",
          "adopt_threshold", "adopt", "detectable",
          "fit_s", "eval_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def base_budget(method):
    b = sweep.DIA_BUDGET if method == "dianetti" else sweep.RL_BUDGET
    return dict(b)


def doubled_budget(method):
    b = base_budget(method)
    b["n_iters"] = 2 * b["n_iters"]                # batch held
    return b


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
    print("ARM: doubled iterations, batch held; annealing ON, schedules proportional to n_iters")
    print(RULE)
    print("CAVEAT: single training seed paired with the sweep's; 0.06 covers Monte Carlo "
          "noise, not seed noise (Dai bimodal at d = 2 in the seed arm)")
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
        seed = sweep.cell_seed(method, d, S0)          # the sweep's own seed
        setting = dict(sweep.RL_SETTINGS[method])
        budget = doubled_budget(method)
        base = base_budget(method)
        arm = f"doubled iterations ({budget['n_iters']}/{budget['batch_size']})"
        params = sweep.make_params(d, seed, {**setting, **budget})
        assert params.lambda_start is None             # annealing as in the sweep
        assert params.n_iters == 2 * base["n_iters"]
        assert params.batch_size == base["batch_size"]
        torch.manual_seed(seed)
        sweep._sync(); t0 = time.perf_counter()
        nets, diag = MODS[method].fit(S0, params, payoff_fn=payoffs.max_call, device=DEVICE)
        sweep._sync(); fit_s = time.perf_counter() - t0
        realised = (f"{diag['anneal']}; lr {params.lr:g} cosine to {params.lr / 20:g} "
                    f"over {params.n_iters} iters")

        n_rand = min(sweep.N_EVAL, sweep.RL_EVAL_CAP) if d >= 50 else sweep.N_EVAL
        ev = randomised_value(method, nets, S0, params, payoffs.max_call, DEVICE,
                              seed=seed + sweep.OFF_RAND, n_paths=n_rand,
                              threshold=sweep.E_INV)
        if method == "dianetti":
            policy = FuelLevelPolicy(params, payoffs.max_call, nets, threshold=sweep.E_INV)
        else:
            policy = GreedyValuePolicy(params, payoffs.max_call, nets)

        NETS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"method": method, "d": d, "S0": S0, "seed": seed,
                    "budget_arm": arm, "realised_schedule": realised,
                    "params": dataclasses.asdict(params),
                    "nets": sweep._state(nets)},
                   NETS_DIR / f"{method}_budget2x_d{d}_S{S0:.0f}.pt")

        cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
        c = cert.diagnostics
        lo, hi = c["ci95"]

        sr = srows.get(key)
        sweep_L0 = float(sr["L0"]) if sr else float("nan")
        sweep_se = float(sr["L0_stderr"]) if sr else float("nan")
        sweep_gap = float(sr["gap"]) if sr else float("nan")
        dL0 = c["lower"] - sweep_L0
        dL0_se = math.sqrt(c["lower_stderr"] ** 2 + sweep_se ** 2)
        threshold = max(GAP_FLOOR, 0.5 * sweep_gap)
        adopt = c["gap"] <= threshold
        detectable = dL0 > DETECT

        print(f"{method:9s} d={d:<3d} S0={S0:.0f} seed={seed}  {arm}\n"
              f"      schedule: {realised}\n"
              f"      L0={c['lower']:.4f} ({c['lower_stderr']:.4f})  U0={c['upper']:.4f}  "
              f"gap={c['gap']:.4f}  CI=[{lo:.4f}, {hi:.4f}]  rand={ev['value']:.4f}\n"
              f"      sweep ({base['n_iters']}/{base['batch_size']}): L0={sweep_L0:.4f} "
              f"({sweep_se:.4f})  gap={sweep_gap:.4f}\n"
              f"      gap(2B)={c['gap']:.4f} vs threshold max({GAP_FLOOR}, gap(B)/2)="
              f"{threshold:.4f}  ADOPT: {adopt}\n"
              f"      dL0={dL0:+.4f} (combined s.e. {dL0_se:.4f})  detectable(>{DETECT}): {detectable}\n"
              f"      fit {fit_s:.0f}s  eval {ev['eval_s']:.0f}s  cert {cert.runtime:.0f}s")

        append_row({
            "method": method, "d": d, "S0": S0,
            "budget_arm": arm, "realised_schedule": realised,
            "settings": ", ".join(f"{k}={v:g}" for k, v in setting.items()),
            "budget": f"{budget['n_iters']}/{budget['batch_size']}",
            "seed": seed,
            "L0": c["lower"], "L0_stderr": c["lower_stderr"],
            "delta": c["delta"], "delta_stderr": c["delta_stderr"],
            "U0": c["upper"], "gap": c["gap"], "ci_lo": lo, "ci_hi": hi,
            "randomized": ev["value"], "randomized_stderr": ev["mc_stderr"],
            "own_greedy": ev["greedy_value"],
            "sweep_L0": sweep_L0, "sweep_L0_stderr": sweep_se, "sweep_gap": sweep_gap,
            "dL0": dL0, "dL0_se": dL0_se,
            "adopt_threshold": threshold, "adopt": adopt, "detectable": detectable,
            "fit_s": round(fit_s, 1), "eval_s": round(ev["eval_s"], 1),
            "cert_s": round(cert.runtime, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
