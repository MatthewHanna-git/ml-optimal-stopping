"""Training budget of DOS, RDBDP and NN-LSM at d = 20.

At S0 = 100, each method is trained at its d = 10 budget and at double it:

    method    base           doubled
    becker    1500 / date    3000 / date
    hure      1500 / date    3000 / date
    nn_lsm    3000 / date    6000 / date

The doubled budget is adopted for both d = 20 and d = 50 if its gap is at
most max(0.15, half the base gap). LSM, which has no budget, runs once for
comparison. Becker et al. (2019, Table 1) give the interval
[51.549, 51.803] for this cell.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import lsm
from src.methods import nn_lsm, becker, hure
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 20
S0 = 100.0
N_EVAL = 2 ** 21
N_TRAIN = 2 ** 18
BATCH = 8192
LR = 1e-2
AB = dict(n_outer=1500, n_inner=10000)
PUB = "Becker 2019 d=20 S0=100: CI [51.549, 51.803]"

# (method label, fit callable, base steps/date, doubled steps/date)
ARMS = [
    ("becker", 1500, 3000),
    ("hure", 1500, 3000),
    ("nn_lsm", 3000, 6000),
]
# ----------------------------------------------------------------------------


def make_params(spd, seed):
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=D, n_steps=9, n_paths=N_EVAL, n_iters=spd * 8,
                  batch_size=BATCH, lr=LR, seed=seed, **AB)


def run_cell(name, spd, seed):
    params = make_params(spd, seed)
    t0 = time.perf_counter()
    if name == "becker":
        policy, _ = becker.fit(S0, params, device=DEVICE)
    elif name == "hure":
        policy, _ = hure.fit(S0, params, device=DEVICE)
    else:
        policy, _, _ = nn_lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
    t_fit = time.perf_counter() - t0
    cert = certify(S0, params, policy, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  {spd}/date: L0 {c['lower']:9.4f} ({c['lower_stderr']:.4f})   "
          f"U0 {c['upper']:9.4f}   gap {c['gap']:.4f}   "
          f"CI [{lo:.4f}, {hi:.4f}]   (fit {t_fit:.0f}s, cert {cert.runtime:.0f}s)")
    return c["gap"]


def main():
    print(f"device: {DEVICE}")
    print(f"anchor: {PUB}")
    t_all = time.perf_counter()

    print("=" * 72)
    print(f"lsm, d={D}, S0={S0:.0f}  —  context cell (no budget lever)")
    print("=" * 72)
    params = make_params(1, 8020)
    t0 = time.perf_counter()
    policy, _, _ = lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
    t_fit = time.perf_counter() - t0
    cert = certify(S0, params, policy, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  L0 {c['lower']:9.4f} ({c['lower_stderr']:.4f})   U0 {c['upper']:9.4f}   "
          f"gap {c['gap']:.4f}   CI [{lo:.4f}, {hi:.4f}]   "
          f"(fit {t_fit:.0f}s, cert {cert.runtime:.0f}s)")

    decisions = []
    for i, (name, base, doubled) in enumerate(ARMS):
        print("=" * 72)
        print(f"{name}, d={D}, S0={S0:.0f}  —  {base}/date vs {doubled}/date")
        print("=" * 72)
        g_base = run_cell(name, base, 8100 + 10 * i)
        g_dbl = run_cell(name, doubled, 8100 + 10 * i + 1)
        thresh = max(0.15, 0.5 * g_base)
        adopt = g_dbl <= thresh
        print(f"  -> {'ADOPT ' + str(doubled) + '/date for d in {20, 50}' if adopt else 'KEEP ' + str(base) + '/date'}"
              f"   (rule: {g_dbl:.4f} vs max(0.15, {0.5 * g_base:.4f}))")
        decisions.append((name, g_base, g_dbl, doubled if adopt else base))

    print("=" * 72)
    print("EXTENSION BUDGET DECISIONS (d = 20 and 50):")
    print(f"  lsm      : n_train 2^18 (no lever)")
    for name, g_base, g_dbl, spd in decisions:
        print(f"  {name:9s}: gap {g_base:.3f} -> {g_dbl:.3f}   use {spd}/date")
    print(f"\ntotal: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
