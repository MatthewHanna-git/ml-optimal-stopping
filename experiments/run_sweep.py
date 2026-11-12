"""The dimension sweep: eight methods across d in {2, 3, 5, 10, 20, 50} and
S0 in {90, 100, 110}, 144 cells.

The contract is the symmetric Bermudan max-call of Becker et al. (2019,
Table 1): K = 100, r = 5%, delta = 10%, sigma = 20%, independent assets,
T = 3, 9 exercise dates. Each cell trains once and certifies the executed
policy by Andersen-Broadie (2^21 lower-bound paths, 1500 outer paths, 10^4
inner paths). For the RL methods the cell also records the value of the
randomised policy under the same conventions (rl_sweep_eval), and for
Dianetti the certified rule at both e^{-1} and 1/2.

Settings:

  lsm       polynomial basis, 2^18 training paths
  nn_lsm    400 steps per date (d <= 3), 1500 (d = 5), 3000 (d >= 10)
  becker    400 steps per date (d <= 5), 1500 (d = 10), 3000 (d >= 20)
  hure      400 steps per date (d <= 3), 1500 (d >= 5)
  dong      lambda = 1e-2,               6000 iterations of 2048 paths
  dai       lambda = 0.1, K = 40,        6000 / 2048
  chee      lambda = 1e-2 (n = 100),     6000 / 2048
  dianetti  lambda = 1e-2,               6000 / 2048, lr 5e-3

Seeds: cell c = 0, ..., 143, in the order method, d, S0, trains from
s = 10^6 + 10^4 c; the randomised evaluation draws from s + 3 and the
certification from s + 7919 and s + 7920. check_seeds() asserts at startup
that no two streams coincide.

Each completed cell is appended to results/sweep_results.csv and its
networks saved to results/nets/; on restart, recorded cells are skipped.
RUN_METHODS and RUN_DIMS split the run without changing any cell's seed.

Usage:  python experiments/run_sweep.py            # the sweep, about 9 h
        python experiments/run_sweep.py --smoke    # a tiny test run
"""

import csv
import dataclasses
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src import payoffs
from src.reference import binomial2d, lsm
from src.methods import nn_lsm, becker, hure, dong, dai, chee, dianetti
from src.methods.rl_policies import GreedyValuePolicy, FuelLevelPolicy
from src.methods.rl_sweep_eval import randomised_value
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv

METHODS = ["lsm", "nn_lsm", "becker", "hure", "dong", "dai", "chee", "dianetti"]
DIMS = [2, 3, 5, 10, 20, 50]
S0_LIST = (90.0, 100.0, 110.0)
RUN_METHODS = set(METHODS)                            # edit to split the run
RUN_DIMS = set(DIMS)                                  # edit to split the run

RESULTS = Path(__file__).resolve().parents[1] / "results"
CSV_PATH = RESULTS / ("smoke_sweep.csv" if SMOKE else "sweep_results.csv")
NETS_DIR = RESULTS / ("nets_smoke" if SMOKE else "nets")

N_EVAL = 2 ** 21
N_TRAIN = 2 ** 18            # lsm / nn_lsm regression paths
BATCH_ML = 8192
LR_ML = 1e-2
AB = dict(n_outer=1500, n_inner=10000)

SPD = {  # steps per date by (method, d)
    "nn_lsm": {2: 400, 3: 400, 5: 1500, 10: 3000, 20: 3000, 50: 3000},
    "becker": {2: 400, 3: 400, 5: 400, 10: 1500, 20: 3000, 50: 3000},
    "hure": {2: 400, 3: 400, 5: 1500, 10: 1500, 20: 1500, 50: 1500},
}
RL_EVAL_CAP = 2 ** 19        # randomised-evaluation paths at d = 50, for memory
RL_BUDGET = dict(n_iters=6000, batch_size=2048)
RL_SETTINGS = {
    "dong": dict(lambda_=1e-2),
    "dai": dict(lambda_=0.1, penalty=40.0),
    "chee": dict(lambda_=1e-2),
    "dianetti": dict(lambda_=1e-2),
}
DIA_BUDGET = dict(n_iters=6000, batch_size=2048, lr=5e-3)

SEED_BLOCK_BASE = 1_000_000
SEED_BLOCK = 10_000
OFF_RAND, OFF_LOWER, OFF_DUAL = 3, 7919, 7920    # certify() owns 7919/7920
LEGACY_SEED_MAX = 70_000     # every validation and calibration seed lies below this

E_INV = math.exp(-1.0)

if SMOKE:                    # tiny budgets, same code path
    DIMS = [2, 5]
    RUN_DIMS = set(DIMS)
    N_EVAL, N_TRAIN, BATCH_ML = 2 ** 12, 2 ** 12, 256
    AB = dict(n_outer=16, n_inner=64)
    SPD = {m: {d: 4 for d in [2, 3, 5, 10, 20, 50]} for m in SPD}
    RL_BUDGET = dict(n_iters=12, batch_size=128)
    DIA_BUDGET = dict(n_iters=12, batch_size=128, lr=5e-3)
    RL_EVAL_CAP = 2 ** 11

PUB = {  # published values, printed for context only (not written to the CSV)
    (2, 90.0): "2019 CI [8.060, 8.081], binomial 8.075",
    (2, 100.0): "2019 CI [13.880, 13.910], binomial 13.902",
    (2, 110.0): "2019 CI [21.336, 21.354], binomial 21.345",
    (3, 90.0): "2019 CI [11.276, 11.290], binomial 11.29",
    (3, 100.0): "2019 CI [18.673, 18.699], binomial 18.69",
    (3, 110.0): "2019 CI [27.545, 27.591], binomial 27.58",
    (5, 90.0): "2019 CI [16.633, 16.648], BC [16.620, 16.653]",
    (5, 100.0): "2019 CI [26.138, 26.174], BC [26.115, 26.164]",
    (5, 110.0): "2019 CI [36.745, 36.789], BC [36.710, 36.798]",
    (10, 90.0): "2019 CI [26.189, 26.289]",
    (10, 100.0): "2019 CI [38.300, 38.367]",
    (10, 110.0): "2019 CI [50.834, 50.937]",
    (20, 90.0): "2019 CI [37.681, 37.942]",
    (20, 100.0): "2019 CI [51.549, 51.803]",
    (20, 110.0): "2019 CI [65.470, 65.812]",
    (50, 90.0): "2019 CI [53.883, 54.266]",
    (50, 100.0): "2019 CI [69.560, 69.945], Lapeyre T10/11 ~68.6-69.7",
    (50, 110.0): "2019 CI [85.204, 85.763]",
}

FIELDS = ["method", "d", "S0", "settings", "budget", "cell", "seed",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap",
          "ci_lo", "ci_hi", "tree_ref",
          "randomized", "randomized_stderr", "rand_paths", "own_greedy",
          "L0_half", "L0_half_stderr", "gap_half", "ci_half_lo", "ci_half_hi",
          "fit_s", "eval_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------

_FULL_DIMS = [2, 3, 5, 10, 20, 50]      # the cell index runs over the full grid


def cell_index(method, d, S0):
    """c = 0..143 in (method, d, S0) order, whatever RUN_METHODS and RUN_DIMS."""
    return (METHODS.index(method) * len(_FULL_DIMS) * len(S0_LIST)
            + _FULL_DIMS.index(d) * len(S0_LIST) + list(S0_LIST).index(S0))


def cell_seed(method, d, S0):
    return SEED_BLOCK_BASE + SEED_BLOCK * cell_index(method, d, S0)


def check_seeds():
    """Every stream has its own seed, above every validation and calibration seed."""
    streams = {}
    for m in METHODS:
        for d in _FULL_DIMS:
            for S0 in S0_LIST:
                s = cell_seed(m, d, S0)
                offs = [0, OFF_LOWER, OFF_DUAL] + ([OFF_RAND] if m in RL_SETTINGS else [])
                for o in offs:
                    streams.setdefault(s + o, []).append((m, d, S0, o))
    n_cells = len(METHODS) * len(_FULL_DIMS) * len(S0_LIST)
    n_streams = sum(len(v) for v in streams.values())
    dups = {k: v for k, v in streams.items() if len(v) > 1}
    lo, hi = min(streams), max(streams)
    assert n_cells == 144, n_cells
    assert not dups, f"seed collisions: {dups}"
    assert max(OFF_RAND, OFF_LOWER, OFF_DUAL) < SEED_BLOCK
    assert lo > LEGACY_SEED_MAX
    idx = sorted(cell_index(m, d, S0) for m in METHODS for d in _FULL_DIMS
                 for S0 in S0_LIST)
    assert idx == list(range(144))
    print(f"seed check: {n_cells} cells, {n_streams} streams "
          f"(train / lower / dual, + randomised eval for 72 RL cells), "
          f"{len(streams)} distinct seeds, 0 collisions; "
          f"range [{lo}, {hi}], all > {LEGACY_SEED_MAX} "
          f"(the validation and calibration seeds)")


def make_params(d, seed, extra):
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=d, n_steps=9, tree_substeps=100, n_paths=N_EVAL,
                  seed=seed, **extra, **AB)


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _state(nets):
    """Portable state for saving: dict-of-state-dicts keyed by date."""
    if isinstance(nets, dict):
        return {int(k): (v.state_dict() if hasattr(v, "state_dict") else v)
                for k, v in nets.items()}
    return {k: n.state_dict() for k, n in enumerate(nets)}


def build_cell(method, d, S0):
    """Train once (fit_s = training only). Returns
    (params, policy, extras, settings str, budget str, fit_s, saved state)."""
    seed = cell_seed(method, d, S0)
    torch.manual_seed(seed)     # reproducible network initialisation
    extras = {}
    if method == "lsm":
        params = make_params(d, seed, {})
        _sync(); t0 = time.perf_counter()
        policy, v_in, _ = lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
        _sync(); fit_s = time.perf_counter() - t0
        return (params, policy, extras, "poly basis",
                f"n_train=2^{N_TRAIN.bit_length() - 1}", fit_s,
                {"coeffs": {int(k): v for k, v in policy.coeffs.items()}})
    if method in ("nn_lsm", "becker", "hure"):
        spd = SPD[method][d]
        params = make_params(d, seed, dict(n_iters=spd * 8,
                                           batch_size=BATCH_ML, lr=LR_ML))
        _sync(); t0 = time.perf_counter()
        if method == "nn_lsm":
            policy, v_in, _ = nn_lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
            label = "ValueNet regressor"
        elif method == "becker":
            policy, _ = becker.fit(S0, params, device=DEVICE)
            label = "2019 recursive DOS"
        else:
            policy, _ = hure.fit(S0, params, device=DEVICE)
            label = "RDBDP"
        _sync(); fit_s = time.perf_counter() - t0
        return (params, policy, extras, label, f"{spd}/date", fit_s,
                {"nets": _state(policy.nets)})

    # RL methods
    mod = {"dong": dong, "dai": dai, "chee": chee, "dianetti": dianetti}[method]
    budget = DIA_BUDGET if method == "dianetti" else RL_BUDGET
    params = make_params(d, seed, {**RL_SETTINGS[method], **budget})
    _sync(); t0 = time.perf_counter()
    nets, _ = mod.fit(S0, params, payoff_fn=payoffs.max_call, device=DEVICE)
    _sync(); fit_s = time.perf_counter() - t0

    # Randomised value under the certification's conventions, from seed s + 3;
    # 2^19 paths at d = 50 for memory (the certified L0 keeps 2^21).
    n_rand = min(N_EVAL, RL_EVAL_CAP) if d >= 50 else N_EVAL
    ev = randomised_value(method, nets, S0, params, payoffs.max_call, DEVICE,
                          seed=seed + OFF_RAND, n_paths=n_rand, threshold=E_INV)
    if ev["otm_stop_frac"] != 0.0:
        print(f"  WARNING: {method} randomised eval stopped at g=0 "
              f"(frac {ev['otm_stop_frac']:.2e}) — guard failed")
    if method == "dianetti":
        policy = FuelLevelPolicy(params, payoffs.max_call, nets, threshold=E_INV)
    else:
        policy = GreedyValuePolicy(params, payoffs.max_call, nets)
    extras = {"randomized": ev["value"], "randomized_stderr": ev["mc_stderr"],
              "rand_paths": ev["n_paths"], "own_greedy": ev["greedy_value"],
              "eval_s": ev["eval_s"], "nets": nets}
    setting = ", ".join(f"{k}={v:g}" for k, v in RL_SETTINGS[method].items())
    blabel = f"{budget['n_iters']}/{budget['batch_size']}"
    return params, policy, extras, setting, blabel, fit_s, {"nets": _state(nets)}


def load_done():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as f:
        return {(r["method"], int(r["d"]), float(r["S0"]))
                for r in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def save_nets(method, d, S0, params, setting, blabel, state):
    NETS_DIR.mkdir(parents=True, exist_ok=True)
    path = NETS_DIR / f"{method}_d{d}_S{S0:.0f}.pt"
    torch.save({"method": method, "d": d, "S0": S0, "seed": params.seed,
                "cell": cell_index(method, d, S0), "settings": setting,
                "budget": blabel, "params": dataclasses.asdict(params),
                "arch": params.arch, **state}, path)


def env_banner():
    print(f"device: {DEVICE}   torch {torch.__version__}"
          + (f"   CUDA {torch.version.cuda}" if DEVICE == "cuda" else ""))
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu: {p.name}, {p.total_memory / 2 ** 30:.1f} GiB")
    try:
        import subprocess
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=RESULTS.parent, capture_output=True, text=True)
        if h.returncode == 0:
            print(f"commit: {h.stdout.strip()}")
    except Exception:
        pass
    print(f"csv: {CSV_PATH}\nnets: {NETS_DIR}")
    if SMOKE:
        print("*** SMOKE MODE: tiny budgets, numbers are meaningless ***")


def main():
    env_banner()
    check_seeds()
    done = load_done()
    if done:
        print(f"resuming: {len(done)} cells already recorded, will be skipped")
    t_all = time.perf_counter()

    tree_refs = {}
    if 2 in RUN_DIMS:
        p = make_params(2, 0, {})
        for S0 in S0_LIST:
            tree_refs[S0] = binomial2d.price(S0, p).value
        print("binomial2d refs (d=2): "
              + "  ".join(f"{s:.0f}: {v:.4f}" for s, v in tree_refs.items()))

    n_run = 0
    for method in METHODS:
        if method not in RUN_METHODS:
            continue
        for d in DIMS:
            if d not in RUN_DIMS:
                continue
            for S0 in S0_LIST:
                key = (method, d, S0)
                if key in done:
                    print(f"skip (done): {method} d={d} S0={S0:.0f}")
                    continue
                (params, policy, extras, setting, blabel,
                 fit_s, state) = build_cell(method, d, S0)
                save_nets(method, d, S0, params, setting, blabel, state)

                cert = certify(S0, params, policy,
                               payoff_fn=payoffs.max_call, device=DEVICE)
                c = cert.diagnostics
                lo, hi = c["ci95"]
                ref = tree_refs.get(S0) if d == 2 else None

                half = {}
                if method == "dianetti":        # the 1/2 rule, same networks and streams
                    pol_half = FuelLevelPolicy(params, payoffs.max_call,
                                               extras["nets"], threshold=0.5)
                    ch = certify(S0, params, pol_half,
                                 payoff_fn=payoffs.max_call, device=DEVICE).diagnostics
                    half = {"L0_half": ch["lower"], "L0_half_stderr": ch["lower_stderr"],
                            "gap_half": ch["gap"], "ci_half_lo": ch["ci95"][0],
                            "ci_half_hi": ch["ci95"][1]}

                line = (f"[{cell_index(method, d, S0):3d}] {method:9s} d={d:<3d} "
                        f"S0={S0:<4.0f} seed={params.seed} "
                        f"L0={c['lower']:9.4f} ({c['lower_stderr']:.4f})  "
                        f"U0={c['upper']:9.4f}  gap={c['gap']:.4f}  "
                        f"CI=[{lo:.4f}, {hi:.4f}]")
                if ref is not None:
                    line += f"  tree={ref:.4f} {'IN' if lo <= ref <= hi else '*** OUT ***'}"
                if extras.get("randomized") is not None:
                    line += f"  rand={extras['randomized']:.4f}"
                if half:
                    line += (f"\n{'':6s}dianetti @1/2: L0={half['L0_half']:.4f} "
                             f"gap={half['gap_half']:.4f} "
                             f"CI=[{half['ci_half_lo']:.4f}, {half['ci_half_hi']:.4f}]")
                line += f"\n{'':6s}fit {fit_s:.0f}s"
                if extras.get("eval_s") is not None:
                    line += f"  eval {extras['eval_s']:.0f}s"
                line += f"  cert {cert.runtime:.0f}s"
                print(line)
                if (d, S0) in PUB:
                    print(f"{'':6s}pub: {PUB[(d, S0)]}")

                append_row({
                    "method": method, "d": d, "S0": S0, "settings": setting,
                    "budget": blabel, "cell": cell_index(method, d, S0),
                    "seed": params.seed,
                    "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                    "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                    "U0": c["upper"], "gap": c["gap"],
                    "ci_lo": lo, "ci_hi": hi,
                    "tree_ref": "" if ref is None else ref,
                    "randomized": extras.get("randomized", ""),
                    "randomized_stderr": extras.get("randomized_stderr", ""),
                    "rand_paths": extras.get("rand_paths", ""),
                    "own_greedy": extras.get("own_greedy", ""),
                    **{k: half.get(k, "") for k in
                       ("L0_half", "L0_half_stderr", "gap_half", "ci_half_lo", "ci_half_hi")},
                    "fit_s": round(fit_s, 1),
                    "eval_s": "" if extras.get("eval_s") is None else round(extras["eval_s"], 1),
                    "cert_s": round(cert.runtime, 1),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                n_run += 1

    print(f"\ncompleted {n_run} new cells "
          f"({len(done)} skipped); total {time.perf_counter() - t_all:.0f}s")
    print(f"results: {CSV_PATH}")


if __name__ == "__main__":
    main()
