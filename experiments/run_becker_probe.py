"""Two follow-up checks on DOS.

A: validity of the interval at d = 3, S0 = 90. The cell is certified four
   times, for three DOS seeds and one LSM policy, at four times the outer
   paths. If at least two intervals sit wholly below the published interval,
   the discrepancy is attributed to the certification rather than to chance.

B: budget and learning-rate schedule at d = 10, S0 = 100. DOS is trained
   with the cosine schedule at 400, 1500 and 3000 steps per date and the
   paper's piecewise schedule at 400 and 1500. The cheapest arm whose policy
   value reaches B_TARGET is adopted for d >= 10 and certified.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import lsm
from src.methods import becker
from src.duality import certify
from src.duality.andersen_broadie import _policy_payoffs
from src import payoffs

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_PROBE_A = True
RUN_PROBE_B = True

BATCH = 8192
LR = 1e-2
N_EVAL = 2 ** 21

# Probe A
A_SEEDS = (31090, 41090, 51090)     # fresh DOS seeds for the d=3 S0=90 cell
A_SPD = 400
A_AB = dict(n_outer=6000, n_inner=10000)
A_THRESHOLD = 11.276                 # published 2019 CI lower edge
PUB_D3_90 = "published 2019: L 11.290  U 11.283  CI [11.276, 11.290]  binomial 11.29 (~0.015 err)"

# Probe B
B_ARMS = (("cosine", 400), ("cosine", 1500), ("cosine", 3000),
          ("paper", 400), ("paper", 1500))
B_TARGET = 38.30                     # 2019 L 38.321 minus 0.02
B_AB = dict(n_outer=1500, n_inner=10000)
PUB_D10_100 = "published 2019: L 38.321  U 38.353  CI [38.300, 38.367];  2021 mean 38.272"
# ----------------------------------------------------------------------------


def d3_params(seed):
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=3, n_steps=9, n_paths=N_EVAL, n_iters=A_SPD * 8,
                  batch_size=BATCH, lr=LR, seed=seed, **A_AB)


def d10_params(spd, seed):
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=10, n_steps=9, n_paths=N_EVAL, n_iters=spd * 8,
                  batch_size=BATCH, lr=LR, seed=seed, **B_AB)


def report_cert(cert):
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"    L0 {c['lower']:9.4f} ({c['lower_stderr']:.4f})   "
          f"U0 {c['upper']:9.4f} (delta {c['delta']:.4f} +- {c['delta_stderr']:.4f})")
    print(f"    CI [{lo:.4f}, {hi:.4f}]   gap {c['gap']:.4f}   ({cert.runtime:.0f}s)")
    return hi


def probe_a():
    print("=" * 72)
    print(f"PROBE A: d=3 S0=90 — anomaly CONFIRMED if >=2 of 4 CI-tops < {A_THRESHOLD}")
    print(f"  {PUB_D3_90}")
    print("=" * 72)
    below = 0
    for seed in A_SEEDS:
        params = d3_params(seed)
        t0 = time.perf_counter()
        policy, _ = becker.fit(90.0, params, device=DEVICE)
        print(f"\n  DOS seed {seed}  (fit {time.perf_counter() - t0:.0f}s)")
        hi = report_cert(certify(90.0, params, policy, device=DEVICE))
        below += hi < A_THRESHOLD
    params = d3_params(61090)
    t0 = time.perf_counter()
    policy, _, _ = lsm.fit(90.0, params, device=DEVICE, n_train=2 ** 18)
    print(f"\n  LSM (different policy family)  (fit {time.perf_counter() - t0:.0f}s)")
    hi = report_cert(certify(90.0, params, policy, device=DEVICE))
    below += hi < A_THRESHOLD
    print(f"\n  CI-tops below {A_THRESHOLD}: {below} of 4")
    if below >= 2:
        print("  -> discrepancy confirmed: the certification needs checking at d = 3")
    else:
        print("  -> not confirmed: consistent with chance")


def probe_b():
    print("=" * 72)
    print(f"PROBE B: d=10 S0=100 — adopt cheapest arm with L >= {B_TARGET}")
    print(f"  {PUB_D10_100}")
    print("=" * 72)
    chosen = None
    for i, (sched, spd) in enumerate(B_ARMS):
        params = d10_params(spd, seed=71000 + i)
        t0 = time.perf_counter()
        policy, _ = becker.fit(100.0, params, device=DEVICE, lr_schedule=sched)
        gen = torch.Generator(device=DEVICE).manual_seed(72000 + i)
        pay = _policy_payoffs(100.0, params, policy, payoffs.max_call,
                              N_EVAL, DEVICE, gen)
        L, se = pay.mean().item(), (pay.std() / (N_EVAL ** 0.5)).item()
        ok = L >= B_TARGET
        print(f"  {sched:6s} {spd:5d}/date: L = {L:8.4f} (stderr {se:.4f})   "
              f"{'QUALIFIES' if ok else 'short'}   ({time.perf_counter() - t0:.0f}s)")
        if ok and chosen is None:
            chosen = (sched, spd, params, policy)
    if chosen is None:
        print(f"\n  no arm reached {B_TARGET}")
    else:
        sched, spd, params, policy = chosen
        print(f"\n  -> adopted for d >= 10: {sched} schedule, {spd} steps/date. "
              f"Certifying that arm:")
        report_cert(certify(100.0, params, policy, device=DEVICE))


def main():
    print(f"device: {DEVICE}")
    t0 = time.perf_counter()
    if RUN_PROBE_A:
        probe_a()
    if RUN_PROBE_B:
        probe_b()
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
