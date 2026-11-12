"""Training budget of NN-LSM at d = 5 and d = 10.

  (a) d = 5,  S0 = 110 at 1500 steps per date (base 400)
  (b) d = 10, S0 = 100 at 3000 steps per date (base 1500)

The larger budget is adopted if the new gap is at most
max(0.15, half the gap at the base budget). The training set stays at
N_TRAIN = 2^18 paths.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.methods import nn_lsm
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVAL = 2 ** 21
N_TRAIN = 2 ** 18
BATCH = 8192
LR = 1e-2
AB = dict(n_outer=1500, n_inner=10000)

# (dim, S0, steps/date, prior gap at prior budget, prior budget label)
ARMS = [
    (5, 110.0, 1500, 0.3041, "400/date"),
    (10, 100.0, 3000, 0.4171, "1500/date"),
]
# ----------------------------------------------------------------------------


def main():
    print(f"device: {DEVICE}")
    t_all = time.perf_counter()
    for d, S0, spd, prior_gap, prior_label in ARMS:
        params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                        dim=d, n_steps=9, n_paths=N_EVAL, n_iters=spd * 8,
                        batch_size=BATCH, lr=LR,
                        seed=9000 + 100 * d + int(S0), **AB)
        thresh = max(0.15, 0.5 * prior_gap)
        print("=" * 72)
        print(f"nn-lsm, d={d}, S0={S0:.0f} at {spd}/date   "
              f"(prior gap {prior_gap} at {prior_label}; threshold {thresh:.3f})")
        print("=" * 72)
        t0 = time.perf_counter()
        policy, v_in, se_in = nn_lsm.fit(S0, params, device=DEVICE,
                                         n_train=N_TRAIN)
        t_fit = time.perf_counter() - t0
        cert = certify(S0, params, policy, device=DEVICE)
        c = cert.diagnostics
        lo, hi = c["ci95"]
        adopt = c["gap"] <= thresh
        print(f"  L0 (certified)    : {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})")
        print(f"  95% CI            : [{lo:.4f}, {hi:.4f}]")
        print(f"  gap               : {c['gap']:.4f}  (prior {prior_gap})")
        print(f"  fit {t_fit:.0f}s, certify {cert.runtime:.0f}s")
        print(f"  -> {'ADOPT ' + str(spd) + '/date for the sweep at this d' if adopt else 'KEEP base budget at d=' + str(d)}")
    print(f"\ntotal: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
