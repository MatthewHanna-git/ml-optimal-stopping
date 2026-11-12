"""Diagnostic: the seed spread of Dong, Dai, Chee and Dianetti at depth.

The counterpart of run_diag_seed_arm.py for the four RL methods,
on the same cells (d in {10, 50}, S0 = 100) and at the sweep's settings and
budgets (asserted against run_sweep.py at startup): Dong lambda = 1e-2;
Dai lambda = 0.1, K = 40; Chee lambda = 1e-2; Dianetti lambda = 1e-2;
6000 iterations of 2048 paths (Dianetti at lr 5e-3), with the sweep's
annealing. The certified rule is the sweep's: greedy for Dong, Dai and Chee,
the fuel level e^{-1} for Dianetti. The randomised evaluation is not run.

Run k = 0, ..., 39, over (method, d, j) in the order method (dong, dai,
chee, dianetti), d (10, 50), seed j (0, ..., 4), trains from
s = 4 * 10^6 + 10^4 k and is certified as in the sweep, from s + 7919 and
s + 7920. check_seeds() asserts that the 120 streams are distinct from each
other, from the sweep's and from those of run_diag_seed_arm.py.

With the sweep cell, each (method, d) has six draws. The CSV records each
draw's certified interval and whether it overlaps Becker et al. (2019,
Table 1). summarise() prints each method's mean, standard deviation and
range of L0 and gap, and sets the doubled-budget differences of
run_diag_dong_budget.py and run_diag_rl_budget.py against the seed spread
(at d = 20, the larger of the d = 10 and d = 50 spreads).

Before the runs, Dong and Chee at d = 10 are retrained from the sweep's
seeds and their L0 printed beside the sweep's, to confirm the code
reproduces the sweep (REPRO_CHECK); small differences can come from GPU
non-determinism.

Writes results/diag_seed_arm_rl.csv (the eight sweep cells, then one row
per run, resumable) and the networks to results/nets_diag/.

Usage:  python experiments/run_diag_seed_arm_rl.py
        python experiments/run_diag_seed_arm_rl.py --smoke
The smoke mode uses tiny budgets, writes only to a temporary folder, and
runs a second pass to confirm that every run is skipped on resume.
"""

import csv
import dataclasses
import gc
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
from src.methods import dong, dai, chee, dianetti
from src.methods.rl_policies import GreedyValuePolicy, FuelLevelPolicy
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
RESULTS = sweep.RESULTS
SWEEP_CSV = RESULTS / "sweep_results.csv"
DONG_BUDGET_CSV = RESULTS / "diag_dong_budget.csv"
RL_BUDGET_CSV = RESULTS / "diag_rl_budget.csv"
if SMOKE:
    OUT_DIR = Path(tempfile.gettempdir()) / "diag_seed_arm_rl_smoke"
    CSV_PATH = OUT_DIR / "diag_seed_arm_rl.csv"
    NETS_DIR = OUT_DIR / "nets_diag"
else:
    CSV_PATH = RESULTS / "diag_seed_arm_rl.csv"
    NETS_DIR = RESULTS / "nets_diag"

METHODS = ["dong", "dai", "chee", "dianetti"]    # the sweep's order; fixes k
NAME = {"dong": "Dong", "dai": "Dai", "chee": "Chee", "dianetti": "Dianetti"}
MODS = {"dong": dong, "dai": dai, "chee": chee, "dianetti": dianetti}
DIMS = [10, 50]
S0 = 100.0
N_SEEDS = 5
SEED_BASE, SEED_STEP = 4_000_000, 10_000
OFF_LOWER, OFF_DUAL = sweep.OFF_LOWER, sweep.OFF_DUAL       # 7919, 7920
SWEEP_STREAM_RANGE = (1_000_000, 2_437_920)
ML_ARM_STREAM_RANGE = (3_000_000, 3_397_920)
REPRO_CHECK = True
REPRO_CELLS = [("dong", 10), ("chee", 10)]

# The rule
ROBUST_K = 2.0                  # |dL0| > ROBUST_K * sqrt(2) * sd_m
BRACKETED = {20: (10, 50)}      # d without a measured s.d.: larger of these
# Becker, Cheridito and Jentzen (2019), Table 1, S0 = 100: (L_hat, U_hat,
# CI_lo, CI_hi).
BECKER_T1 = {10: (38.321, 38.353, 38.300, 38.367),
             50: (69.582, 69.889, 69.560, 69.945)}

# The sweep's settings, asserted against run_sweep.py at startup
EXPECTED_SETTINGS = {"dong": dict(lambda_=1e-2),
                     "dai": dict(lambda_=0.1, penalty=40.0),
                     "chee": dict(lambda_=1e-2),
                     "dianetti": dict(lambda_=1e-2)}
EXPECTED_RL_BUDGET = dict(n_iters=6000, batch_size=2048)
EXPECTED_DIA_BUDGET = dict(n_iters=6000, batch_size=2048, lr=5e-3)
EXPECTED_CERT = dict(n_paths=2 ** 21, n_outer=1500, n_inner=10000)

# Budgets as run (the sweep's, unless smoke)
RL_BUDGET = dict(sweep.RL_BUDGET)
DIA_BUDGET = dict(sweep.DIA_BUDGET)
SMOKE_CERT = None
if SMOKE:                    # tiny budgets, same code path
    RL_BUDGET = dict(n_iters=12, batch_size=128)
    DIA_BUDGET = dict(n_iters=12, batch_size=128, lr=5e-3)
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


def budget_of(method):
    return DIA_BUDGET if method == "dianetti" else RL_BUDGET


def budget_label(budget):
    return f"{budget['n_iters']}/{budget['batch_size']}"


def setting_label(method):
    return ", ".join(f"{k}={v:g}" for k, v in sweep.RL_SETTINGS[method].items())


def check_config():
    """The sweep's settings, budgets and certification are as the design states."""
    for m, s in EXPECTED_SETTINGS.items():
        assert sweep.RL_SETTINGS[m] == s, f"{m}: sweep has {sweep.RL_SETTINGS[m]}, design {s}"
    assert sweep.RL_BUDGET == EXPECTED_RL_BUDGET, sweep.RL_BUDGET
    assert sweep.DIA_BUDGET == EXPECTED_DIA_BUDGET, sweep.DIA_BUDGET
    assert sweep.N_EVAL == EXPECTED_CERT["n_paths"], sweep.N_EVAL
    assert sweep.AB == dict(n_outer=EXPECTED_CERT["n_outer"],
                            n_inner=EXPECTED_CERT["n_inner"]), sweep.AB
    assert (OFF_LOWER, OFF_DUAL) == (7919, 7920)
    from src.duality import andersen_broadie
    import inspect
    src = inspect.getsource(andersen_broadie.certify)
    assert "seed + 7919" in src and "seed + 7920" in src, "certify() streams moved"
    p = sweep.make_params(10, 1, {**sweep.RL_SETTINGS["dong"], **sweep.RL_BUDGET})
    assert p.lambda_start is None, "annealing default changed (lambda_start)"
    print("config check: settings and budgets from run_sweep.py  "
          + "  ".join(f"{NAME[m]} {setting_label(m)}" for m in METHODS)
          + f"; budget {budget_label(sweep.RL_BUDGET)} (Dianetti lr {sweep.DIA_BUDGET['lr']:g}), "
          f"annealing on; certified rule greedy (Dong, Dai, Chee), fuel level e^-1 "
          f"= {sweep.E_INV:.4f} (Dianetti); certification n=2^{sweep.N_EVAL.bit_length() - 1}, "
          f"n1={sweep.AB['n_outer']}, n2={sweep.AB['n_inner']}, streams s+7919 / s+7920")


def ml_arm_streams():
    """Every stream of run_diag_seed_arm.py, from its seed formula."""
    out = set()
    for k in range(40):
        s = 3_000_000 + 10_000 * k
        out.update((s, s + OFF_LOWER, s + OFF_DUAL))
    return out


def check_seeds():
    """120 streams of the arm: distinct, and disjoint from the sweep's and the ML arm's."""
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
    ml_arm = ml_arm_streams()
    assert (min(ml_arm), max(ml_arm)) == ML_ARM_STREAM_RANGE, (min(ml_arm), max(ml_arm))
    assert not set(streams) & sw and not set(streams) & ml_arm
    for lo, hi in (SWEEP_STREAM_RANGE, ML_ARM_STREAM_RANGE):
        assert all(not (lo <= s <= hi) for s in streams)
    print(f"seed check: 40 runs, 120 streams (train s, lower s+7919, dual s+7920), "
          f"120 distinct; range [{min(streams)}, {max(streams)}]; none in the sweep's "
          f"{len(sw)} streams [{SWEEP_STREAM_RANGE[0]}, {SWEEP_STREAM_RANGE[1]}] "
          f"or the ML arm's {len(ml_arm)} [{ML_ARM_STREAM_RANGE[0]}, {ML_ARM_STREAM_RANGE[1]}]")


def make_params(method, d, seed):
    p = sweep.make_params(d, seed, {**sweep.RL_SETTINGS[method], **budget_of(method)})
    return dataclasses.replace(p, **SMOKE_CERT) if SMOKE else p


def fit_one(method, d, seed):
    """Train and extract as the sweep's build_cell does (no randomised
    evaluation). Returns (params, policy, fit_s, saved state)."""
    torch.manual_seed(seed)
    params = make_params(method, d, seed)
    assert params.lambda_start is None
    sweep._sync(); t0 = time.perf_counter()
    nets, _ = MODS[method].fit(S0, params, payoff_fn=payoffs.max_call, device=DEVICE)
    sweep._sync(); fit_s = time.perf_counter() - t0
    if method == "dianetti":
        policy = FuelLevelPolicy(params, payoffs.max_call, nets, threshold=sweep.E_INV)
    else:
        policy = GreedyValuePolicy(params, payoffs.max_call, nets)
    return params, policy, fit_s, {"nets": sweep._state(nets)}


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_sweep_rows():
    return {(r["method"], int(r["d"]), float(r["S0"])): r for r in load_csv(SWEEP_CSV)}


def load_rows():
    return load_csv(CSV_PATH)


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
    """Dong and Chee at d = 10 from the sweep's own seeds: L0 beside the CSV's."""
    for m, d in REPRO_CELLS:
        r = srows[(m, d, S0)]
        seed = sweep.cell_seed(m, d, S0)
        assert int(r["seed"]) == seed
        params, policy, fit_s, _ = fit_one(m, d, seed)
        c = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE).diagnostics
        diff = c["lower"] - float(r["L0"])
        print(f"reproduction check, {NAME[m]} d={d} S0=100 seed={seed}: L0 now "
              f"{c['lower']:.6f} (s.e. {c['lower_stderr']:.6f}), sweep CSV {float(r['L0']):.6f}; "
              f"difference {diff:+.2e} ({diff / c['lower_stderr']:+.1f} s.e.)  gap now "
              f"{c['gap']:.6f}, CSV {float(r['gap']):.6f}  (fit {fit_s:.0f}s)")
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
            assert r["budget"] == budget_label(EXPECTED_RL_BUDGET), (m, d, r["budget"])
            assert r["settings"] == setting_label(m), (m, d, r["settings"])
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
                params, policy, fit_s, state = fit_one(m, d, seed)
                NETS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"method": m, "d": d, "S0": S0, "seed": seed, "j": j, "k": k,
                            "settings": setting_label(m),
                            "budget": budget_label(budget_of(m)),
                            "params": dataclasses.asdict(params), **state},
                           NETS_DIR / f"{m}_seed{j}_d{d}_S{S0:.0f}.pt")
                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
                c = cert.diagnostics
                lo, hi = c["ci95"]
                peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                        if DEVICE == "cuda" else float("nan"))
                ov = overlaps(lo, hi, d)
                b = BECKER_T1[d]
                print(f"[k={k:02d}] {m:8s} d={d:<3d} j={j} seed={seed}  "
                      f"L0={c['lower']:9.4f} ({c['lower_stderr']:.4f})  U0={c['upper']:9.4f}  "
                      f"gap={c['gap']:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
                      f"Becker [{b[2]:.3f}, {b[3]:.3f}] {'overlap' if ov else '*** NO OVERLAP ***'}\n"
                      f"      fit {fit_s:.0f}s  cert {cert.runtime:.0f}s"
                      + (f"  peak {peak:.2f} GiB" if DEVICE == "cuda" else ""))
                append_row({
                    "method": m, "d": d, "S0": S0, "j": j, "k": k, "source": "this run",
                    "settings": setting_label(m), "budget": budget_label(budget_of(m)),
                    "seed": seed,
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
    sd, complete = {}, {}
    for d in DIMS:
        for m in METHODS:
            rs = draws.get((m, d), [])
            L = [float(r["L0"]) for r in rs]
            G = [float(r["gap"]) for r in rs]
            ov = [r["overlaps_becker"] == "True" for r in rs]
            sd[(m, d)] = statistics.stdev(L) if len(L) > 1 else float("nan")
            complete[(m, d)] = len(rs) == N_SEEDS + 1
            tag = "" if complete[(m, d)] else f"  INCOMPLETE ({len(rs)} draws)"
            if len(L) > 1:
                print(f"  {NAME[m]:8s} d={d:<3d} L0 mean {statistics.mean(L):.4f} s.d. "
                      f"{sd[(m, d)]:.4f} range [{min(L):.4f}, {max(L):.4f}]  "
                      f"gap mean {statistics.mean(G):.4f} s.d. {statistics.stdev(G):.4f} "
                      f"range [{min(G):.4f}, {max(G):.4f}]  "
                      f"all overlap Becker: {all(ov)} ({sum(ov)}/{len(ov)}){tag}")
            else:
                print(f"  {NAME[m]:8s} d={d:<3d} {len(rs)} draw(s){tag}")

    print(f"RULE: budget difference robust iff |dL0| > {ROBUST_K:g} sqrt(2) sd_m "
          f"(d = 20: larger of the d = 10 and d = 50 s.d.)")
    budget_rows = [("dong budget", r) for r in load_csv(DONG_BUDGET_CSV)] + [("rl budget", r) for r in load_csv(RL_BUDGET_CSV)]
    if not budget_rows:
        print("  (diag_dong_budget.csv / diag_rl_budget.csv not found)")
    for run, r in budget_rows:
        m, d, dL0 = r["method"], int(r["d"]), float(r["dL0"])
        if d in DIMS:
            s, note, ok = sd.get((m, d), float("nan")), "measured", complete.get((m, d), False)
        elif d in BRACKETED:
            a, b = BRACKETED[d]
            s = max(sd.get((m, a), float("nan")), sd.get((m, b), float("nan")))
            note = f"bracketed by d={a},{b}"
            ok = complete.get((m, a), False) and complete.get((m, b), False)
        else:
            print(f"  {run} {NAME.get(m, m):8s} d={d:<3d} dL0 {dL0:+.4f}  not assessed "
                  f"(no s.d. measured at this d; dL0 already within the standard errors)")
            continue
        thr = ROBUST_K * 2 ** 0.5 * s
        print(f"  {run} {NAME.get(m, m):8s} d={d:<3d} dL0 {dL0:+.4f}  sd_m {s:.4f} ({note})  "
              f"threshold {thr:.4f}  robust: {abs(dL0) > thr}"
              + ("" if ok else "  INCOMPLETE"))
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
          "a doubled-budget difference is robust iff |dL0| > 2 sqrt(2) sd_m (d = 20 bracketed); "
          "validity: every interval overlaps Becker T1 ([38.300, 38.367] d=10, [69.560, 69.945] d=50)")
    srows = load_sweep_rows()
    for d in DIMS:
        for m in METHODS:
            r = srows[(m, d, S0)]
            print(f"sweep cell {NAME[m]:8s} d={d:<3d} seed={r['seed']} budget {r['budget']}: "
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
