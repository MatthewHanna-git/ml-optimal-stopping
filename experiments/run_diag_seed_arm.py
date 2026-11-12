"""Diagnostic: the seed spread of LSM, NN-LSM, RDBDP and DOS at depth.

Each sweep cell is a single training seed. This retrains the four regression
and deep methods at d in {10, 50}, S0 = 100, from five new seeds each, at the
sweep's settings and budgets (asserted against run_sweep.py at startup):

  d = 10: NN-LSM 3000, RDBDP 1500, DOS 1500 steps per date;
  d = 50: NN-LSM 3000, RDBDP 1500, DOS 3000 steps per date;
  LSM: polynomial basis, 2^18 training paths.

Run k = 0, ..., 39, over (method, d, j) in the order method (lsm, nn_lsm,
becker, hure), d (10, 50), seed j (0, ..., 4), trains from
s = 3 * 10^6 + 10^4 k and is certified as in the sweep, from s + 7919 and
s + 7920 (n = 2^21, n1 = 1500, n2 = 10^4). check_seeds() asserts that the
120 streams are distinct from each other and from the sweep's.

With the sweep cell, each (method, d) has six draws. The CSV records each
draw's certified interval and whether it overlaps Becker et al. (2019,
Table 1). summarise() prints each method's mean, standard deviation and
range of L0 and gap, and, for the thesis's contrasts (class: LSM less
NN-LSM; target: NN-LSM less RDBDP; object: DOS less RDBDP), the sweep
cells' difference against the seed spread.

Before the runs, LSM at d = 10 is retrained from the sweep's seed and its L0
printed beside the sweep's, to confirm the code reproduces the sweep
(REPRO_CHECK).

Writes results/diag_seed_arm.csv (the eight sweep cells, then one row per
run, resumable) and the networks to results/nets_diag/. At d = 50 the lower
bound is generated in chunks of about 1 GiB; peak memory is printed per run.

Usage:  python experiments/run_diag_seed_arm.py
        python experiments/run_diag_seed_arm.py --smoke
The smoke mode uses tiny budgets, writes only to a temporary folder, and
runs a second pass to confirm that every run is skipped on resume.
"""

import csv
import dataclasses
import gc
import inspect
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# Import the sweep with --smoke hidden, so `sweep` always carries the full
# configuration; the smoke overrides are applied below, in this file only.
SMOKE = "--smoke" in sys.argv
_ARGV = sys.argv
sys.argv = [a for a in _ARGV if a != "--smoke"]
import run_sweep as sweep                      # seeds, params, settings
sys.argv = _ARGV
assert not sweep.SMOKE

import torch

from src import payoffs
from src.reference import lsm
from src.methods import nn_lsm, becker, hure
from src.duality import certify
from src.duality import andersen_broadie

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
RESULTS = sweep.RESULTS
SWEEP_CSV = RESULTS / "sweep_results.csv"
if SMOKE:
    OUT_DIR = Path(tempfile.gettempdir()) / "diag_seed_arm_smoke"
    CSV_PATH = OUT_DIR / "diag_seed_arm.csv"
    NETS_DIR = OUT_DIR / "nets_diag"
else:
    CSV_PATH = RESULTS / "diag_seed_arm.csv"
    NETS_DIR = RESULTS / "nets_diag"

METHODS = ["lsm", "nn_lsm", "becker", "hure"]     # the sweep's order; fixes k
NAME = {"lsm": "LSM", "nn_lsm": "NN-LSM", "becker": "DOS", "hure": "RDBDP"}
DIMS = [10, 50]
S0 = 100.0
N_SEEDS = 5
SEED_BASE, SEED_STEP = 3_000_000, 10_000
OFF_LOWER, OFF_DUAL = sweep.OFF_LOWER, sweep.OFF_DUAL       # 7919, 7920
SWEEP_STREAM_RANGE = (1_000_000, 2_437_920)
REPRO_CHECK = True

# The rule
CONTRASTS = [("class", "LSM less NN-LSM", "lsm", "nn_lsm"),
             ("target", "NN-LSM less RDBDP", "nn_lsm", "hure"),
             ("object", "DOS less RDBDP", "becker", "hure")]
ROBUST_K = 2.0
# Becker, Cheridito and Jentzen (2019), Table 1, S0 = 100: (L_hat, U_hat,
# CI_lo, CI_hi).
BECKER_T1 = {10: (38.321, 38.353, 38.300, 38.367),
             50: (69.582, 69.889, 69.560, 69.945)}

# The sweep's settings, asserted against run_sweep.py at startup
EXPECTED_SPD = {("nn_lsm", 10): 3000, ("hure", 10): 1500, ("becker", 10): 1500,
                ("nn_lsm", 50): 3000, ("hure", 50): 1500, ("becker", 50): 3000}
EXPECTED_N_TRAIN = 2 ** 18
EXPECTED_CERT = dict(n_paths=2 ** 21, n_outer=1500, n_inner=10000)

# Budgets as run (the sweep's, unless smoke)
N_TRAIN = sweep.N_TRAIN
BATCH = sweep.BATCH_ML
SPD = {m: dict(sweep.SPD[m]) for m in ("nn_lsm", "becker", "hure")}
SMOKE_CERT = None
if SMOKE:                    # tiny budgets, same code path
    N_TRAIN, BATCH = 2 ** 12, 256
    SPD = {m: {d: 4 for d in DIMS} for m in SPD}
    SMOKE_CERT = dict(n_paths=2 ** 12, n_outer=16, n_inner=64)
    REPRO_CHECK = False

FIELDS = ["method", "d", "S0", "j", "k", "source", "settings", "budget", "seed",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "becker_ci_lo", "becker_ci_hi", "overlaps_becker",
          "peak_mem_gib", "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


def run_index(method, d, j):
    """k = 0..39 over (method, d, j)."""
    return (METHODS.index(method) * len(DIMS) + DIMS.index(d)) * N_SEEDS + j


def run_seed(method, d, j):
    return SEED_BASE + SEED_STEP * run_index(method, d, j)


def check_config():
    """The sweep's budgets and certification parameters are as the design states."""
    for (m, d), spd in EXPECTED_SPD.items():
        assert sweep.SPD[m][d] == spd, f"{m} d={d}: sweep has {sweep.SPD[m][d]}, design {spd}"
    assert sweep.N_TRAIN == EXPECTED_N_TRAIN, sweep.N_TRAIN
    assert sweep.N_EVAL == EXPECTED_CERT["n_paths"], sweep.N_EVAL
    assert sweep.AB == dict(n_outer=EXPECTED_CERT["n_outer"],
                            n_inner=EXPECTED_CERT["n_inner"]), sweep.AB
    assert (OFF_LOWER, OFF_DUAL) == (7919, 7920)
    src = inspect.getsource(andersen_broadie.certify)
    assert "seed + 7919" in src and "seed + 7920" in src, "certify() streams moved"
    print("config check: budgets from run_sweep.py  "
          + "  ".join(f"{NAME[m]} d={d} {sweep.SPD[m][d]}/date"
                      for (m, d) in sorted(EXPECTED_SPD, key=lambda x: (x[1], x[0])))
          + f"  LSM n_train=2^{sweep.N_TRAIN.bit_length() - 1}; "
          f"batch {sweep.BATCH_ML}, lr {sweep.LR_ML:g}, iterations = steps/date x 8; "
          f"certification n=2^{sweep.N_EVAL.bit_length() - 1}, "
          f"n1={sweep.AB['n_outer']}, n2={sweep.AB['n_inner']}, streams s+7919 / s+7920")


def check_seeds():
    """120 streams of the arm: distinct, and disjoint from every sweep stream."""
    streams = {}
    for m in METHODS:
        for d in DIMS:
            for j in range(N_SEEDS):
                s = run_seed(m, d, j)
                for o in (0, OFF_LOWER, OFF_DUAL):
                    streams.setdefault(s + o, []).append((m, d, j, o))
    n_streams = sum(len(v) for v in streams.values())
    ks = sorted(run_index(m, d, j) for m in METHODS for d in DIMS for j in range(N_SEEDS))
    assert ks == list(range(40)), ks
    assert n_streams == 120 and len(streams) == 120, "stream collision inside the arm"
    sw = set()
    for m in sweep.METHODS:
        for d in sweep._FULL_DIMS:
            for s0 in sweep.S0_LIST:
                c = sweep.cell_seed(m, d, s0)
                offs = [0, sweep.OFF_LOWER, sweep.OFF_DUAL]
                offs += [sweep.OFF_RAND] if m in sweep.RL_SETTINGS else []
                sw.update(c + o for o in offs)
    assert (min(sw), max(sw)) == SWEEP_STREAM_RANGE, (min(sw), max(sw))
    lo, hi = SWEEP_STREAM_RANGE
    assert not set(streams) & sw
    assert all(not (lo <= s <= hi) for s in streams)
    print(f"seed check: 40 runs, 120 streams (train s, lower s+7919, dual s+7920), "
          f"120 distinct; range [{min(streams)}, {max(streams)}]; none in the sweep's "
          f"{len(sw)} streams, range [{lo}, {hi}]")


def make_params(d, seed, extra):
    p = sweep.make_params(d, seed, extra)
    return dataclasses.replace(p, **SMOKE_CERT) if SMOKE else p


def fit_one(method, d, seed):
    """Train as the sweep's build_cell does. Returns
    (params, policy, settings, budget, fit_s, saved state)."""
    torch.manual_seed(seed)
    if method == "lsm":
        params = make_params(d, seed, {})
        sweep._sync(); t0 = time.perf_counter()
        policy, _, _ = lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
        sweep._sync(); fit_s = time.perf_counter() - t0
        return (params, policy, "poly basis", f"n_train=2^{N_TRAIN.bit_length() - 1}",
                fit_s, {"coeffs": {int(k): v for k, v in policy.coeffs.items()}})
    spd = SPD[method][d]
    params = make_params(d, seed, dict(n_iters=spd * 8, batch_size=BATCH, lr=sweep.LR_ML))
    sweep._sync(); t0 = time.perf_counter()
    if method == "nn_lsm":
        policy, _, _ = nn_lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
        label = "ValueNet regressor"
    elif method == "becker":
        policy, _ = becker.fit(S0, params, device=DEVICE)
        label = "2019 recursive DOS"
    else:
        policy, _ = hure.fit(S0, params, device=DEVICE)
        label = "RDBDP"
    sweep._sync(); fit_s = time.perf_counter() - t0
    return params, policy, label, f"{spd}/date", fit_s, {"nets": sweep._state(policy.nets)}


def load_sweep_rows():
    rows = {}
    with open(SWEEP_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["method"], int(r["d"]), float(r["S0"]))] = r
    return rows


def load_rows():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def append_row(row):
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def overlaps(lo, hi, d):
    b = BECKER_T1[d]
    return lo <= b[3] and hi >= b[2]


def _free():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def repro_check(srows):
    """LSM at d = 10 from the sweep's own seed: prints L0 beside the CSV's."""
    r = srows[("lsm", 10, S0)]
    seed = sweep.cell_seed("lsm", 10, S0)
    assert int(r["seed"]) == seed
    params, policy, *_ = fit_one("lsm", 10, seed)
    c = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE).diagnostics
    print(f"reproduction check, LSM d=10 S0=100 seed={seed}: L0 now {c['lower']:.6f} "
          f"(s.e. {c['lower_stderr']:.6f}), sweep CSV {float(r['L0']):.6f}; "
          f"difference {c['lower'] - float(r['L0']):+.2e}  gap now {c['gap']:.6f}, "
          f"CSV {float(r['gap']):.6f}")
    del policy
    _free()


def write_sweep_rows(srows, done):
    """The six-draw sets start from the sweep's cells (j and k blank)."""
    for m in METHODS:
        for d in DIMS:
            if (m, d, "") in done:
                continue
            r = srows[(m, d, S0)]
            assert int(r["seed"]) == sweep.cell_seed(m, d, S0)
            exp_budget = (f"n_train=2^{EXPECTED_N_TRAIN.bit_length() - 1}" if m == "lsm"
                          else f"{EXPECTED_SPD[(m, d)]}/date")
            assert r["budget"] == exp_budget, (m, d, r["budget"])
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            append_row({
                "method": m, "d": d, "S0": S0, "j": "", "k": "",
                "source": "sweep_results.csv", "settings": r["settings"],
                "budget": r["budget"], "seed": r["seed"],
                **{f: r[f] for f in ("L0", "L0_stderr", "delta", "delta_stderr", "U0",
                                     "gap", "ci_lo", "ci_hi", "fit_s", "cert_s")},
                "becker_ci_lo": BECKER_T1[d][2], "becker_ci_hi": BECKER_T1[d][3],
                "overlaps_becker": overlaps(lo, hi, d), "peak_mem_gib": "",
                "timestamp": r["timestamp"],
            })


def run_arm():
    """Every run not yet in the CSV. Returns the number of new runs."""
    done = {(r["method"], int(r["d"]), r["j"]) for r in load_rows()}
    n_done = sum(1 for key in done if key[2] != "")
    if n_done:
        print(f"resuming: {n_done} runs already recorded, will be skipped")
    n_new = 0
    for j in range(N_SEEDS):
        for d in DIMS:
            for m in METHODS:
                k = run_index(m, d, j)
                if (m, d, str(j)) in done:
                    print(f"skip (done): [k={k:02d}] {m} d={d} j={j}")
                    continue
                seed = run_seed(m, d, j)
                if DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                params, policy, setting, blabel, fit_s, state = fit_one(m, d, seed)
                NETS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"method": m, "d": d, "S0": S0, "seed": seed, "j": j, "k": k,
                            "settings": setting, "budget": blabel,
                            "params": dataclasses.asdict(params), "arch": params.arch,
                            **state}, NETS_DIR / f"{m}_seed{j}_d{d}_S{S0:.0f}.pt")
                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
                c = cert.diagnostics
                lo, hi = c["ci95"]
                peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                        if DEVICE == "cuda" else float("nan"))
                ov = overlaps(lo, hi, d)
                b = BECKER_T1[d]
                print(f"[k={k:02d}] {m:7s} d={d:<3d} j={j} seed={seed}  "
                      f"L0={c['lower']:9.4f} ({c['lower_stderr']:.4f})  U0={c['upper']:9.4f}  "
                      f"gap={c['gap']:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
                      f"Becker [{b[2]:.3f}, {b[3]:.3f}] {'overlap' if ov else '*** NO OVERLAP ***'}\n"
                      f"      fit {fit_s:.0f}s  cert {cert.runtime:.0f}s"
                      + (f"  peak {peak:.2f} GiB" if DEVICE == "cuda" else ""))
                append_row({
                    "method": m, "d": d, "S0": S0, "j": j, "k": k, "source": "this run",
                    "settings": setting, "budget": blabel, "seed": seed,
                    "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                    "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                    "U0": c["upper"], "gap": c["gap"], "ci_lo": lo, "ci_hi": hi,
                    "becker_ci_lo": b[2], "becker_ci_hi": b[3], "overlaps_becker": ov,
                    "peak_mem_gib": "" if DEVICE != "cuda" else round(peak, 3),
                    "fit_s": round(fit_s, 1), "cert_s": round(cert.runtime, 1),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                n_new += 1
                del policy, state, cert
                _free()
    return n_new


def summarise():
    """Print the summary of whatever the CSV holds."""
    rows = load_rows()
    draws = {}
    for r in rows:
        draws.setdefault((r["method"], int(r["d"])), []).append(r)
    print("=" * 78)
    print("SUMMARY (six draws per (method, d): the sweep cell + five seeds)")
    sd = {}
    for d in DIMS:
        for m in METHODS:
            rs = draws.get((m, d), [])
            L = [float(r["L0"]) for r in rs]
            G = [float(r["gap"]) for r in rs]
            ov = [r["overlaps_becker"] == "True" for r in rs]
            sd[(m, d)] = statistics.stdev(L) if len(L) > 1 else float("nan")
            tag = "" if len(rs) == N_SEEDS + 1 else f"  INCOMPLETE ({len(rs)} draws)"
            if len(G) > 1:
                print(f"  {NAME[m]:7s} d={d:<3d} L0 mean {statistics.mean(L):.4f} s.d. {sd[(m, d)]:.4f}  "
                      f"gap mean {statistics.mean(G):.4f} s.d. {statistics.stdev(G):.4f} "
                      f"range [{min(G):.4f}, {max(G):.4f}]  "
                      f"all overlap Becker: {all(ov)} ({sum(ov)}/{len(ov)}){tag}")
            else:
                print(f"  {NAME[m]:7s} d={d:<3d} {len(rs)} draw(s){tag}")
    print(f"RULE: robust iff |C| > {ROBUST_K:g} sqrt(sd_a^2 + sd_b^2), C from the sweep cells")
    for d in DIMS:
        for name, what, a, b in CONTRASTS:
            ra = [r for r in draws.get((a, d), []) if r["source"] == "sweep_results.csv"]
            rb = [r for r in draws.get((b, d), []) if r["source"] == "sweep_results.csv"]
            if not (ra and rb):
                continue
            C = float(ra[0]["L0"]) - float(rb[0]["L0"])
            se = (sd[(a, d)] ** 2 + sd[(b, d)] ** 2) ** 0.5
            complete = all(len(draws.get((x, d), [])) == N_SEEDS + 1 for x in (a, b))
            print(f"  {name:6s} ({what}) d={d:<3d} C {C:+.4f}  seed s.e. {se:.4f}  "
                  f"|C|/s.e. {abs(C) / se:.1f}  robust: {abs(C) > ROBUST_K * se}"
                  + ("" if complete else "  INCOMPLETE"))
    print("=" * 78)


def main():
    sweep.env_banner()
    print(f"diag csv: {CSV_PATH}\ndiag nets: {NETS_DIR}")
    if SMOKE:
        print("*** SMOKE MODE: tiny budgets, numbers are meaningless; nothing written to results/ ***")
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    check_config()
    check_seeds()
    print("RULE: six draws per (method, d) (sweep cell + five seeds); sd_m = sample s.d. of L0; "
          "contrast robust iff |C| > 2 sqrt(sd_a^2 + sd_b^2), C = difference of the sweep cells; "
          "validity: every interval overlaps Becker T1 ([38.300, 38.367] d=10, [69.560, 69.945] d=50)")
    srows = load_sweep_rows()
    for d in DIMS:
        for m in METHODS:
            r = srows[(m, d, S0)]
            print(f"sweep cell {NAME[m]:7s} d={d:<3d} seed={r['seed']} budget {r['budget']}: "
                  f"L0={float(r['L0']):.4f} ({float(r['L0_stderr']):.4f})  gap={float(r['gap']):.4f}  "
                  f"CI=[{float(r['ci_lo']):.4f}, {float(r['ci_hi']):.4f}]")
    if REPRO_CHECK:
        repro_check(srows)
    t_all = time.perf_counter()
    write_sweep_rows(srows, {(r["method"], int(r["d"]), r["j"]) for r in load_rows()})
    n_new = run_arm()
    print(f"\ncompleted {n_new} new runs; total {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")
    if SMOKE:
        print("smoke: second pass, every run should be skipped")
        assert run_arm() == 0, "resume failed: a run was repeated"
        assert len(load_rows()) == len(METHODS) * len(DIMS) * (N_SEEDS + 1)
        print("smoke: resume OK")
    summarise()


if __name__ == "__main__":
    main()
