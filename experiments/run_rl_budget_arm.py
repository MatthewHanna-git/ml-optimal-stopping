"""Training budget of the RL methods, at d = 2.

Each method reruns its worst cell of run_rl_pilot.py, on the same seed, at
double the budget:

    method    S0     budget              doubled
    dong      110    3000 / 1024         6000 / 2048
    dai        90    3000 / 1024         6000 / 2048
    chee      100    3000 / 1024         6000 / 2048
    dianetti   90    6000 / 2048 (5e-3)  12000 / 4096 (5e-3)

The doubled budget is adopted if the new gap is at most
max(0.15, half the gap at the base budget).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src import payoffs
from src.reference import binomial2d
from src.methods import dong, dai, chee, dianetti
from src.methods.rl_policies import GreedyValuePolicy, FuelMedianPolicy
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVAL = 2 ** 21
AB = dict(n_outer=1500, n_inner=10000)
GAP_TARGET = 0.15

# (module, policy class, worst S0, pilot gap, pilot seed base, settings, doubled budget)
ARMS = [
    ("dong", dong, GreedyValuePolicy, 110.0, 0.1962, 21000,
     dict(lambda_=1e-2), dict(n_iters=6000, batch_size=2048)),
    ("dai", dai, GreedyValuePolicy, 90.0, 0.1870, 22000,
     dict(lambda_=0.1, penalty=40.0), dict(n_iters=6000, batch_size=2048)),
    ("chee", chee, GreedyValuePolicy, 100.0, 0.1753, 23000,
     dict(lambda_=1e-2), dict(n_iters=6000, batch_size=2048)),
    ("dianetti", dianetti, FuelMedianPolicy, 90.0, 0.3808, 24000,
     dict(lambda_=1e-2), dict(n_iters=12000, batch_size=4096, lr=5e-3)),
]
# ----------------------------------------------------------------------------


def main():
    print(f"device: {DEVICE}")
    t_all = time.perf_counter()
    verdicts = []
    for name, mod, Pol, S0, pilot_gap, seed_base, pd, budget in ARMS:
        params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                        dim=2, n_steps=9, tree_substeps=100, n_paths=N_EVAL,
                        seed=seed_base + int(S0), **pd, **budget, **AB)
        ref = binomial2d.price(S0, params).value
        print("=" * 72)
        print(f"{name}, S0 = {S0:.0f}  —  doubled budget {budget}   "
              f"(pilot gap {pilot_gap})")
        print("=" * 72)
        t0 = time.perf_counter()
        nets, _ = mod.fit(S0, params, payoff_fn=payoffs.max_call, device=DEVICE)
        ev = mod.evaluate(nets, S0, params, payoff_fn=payoffs.max_call,
                          device=DEVICE)
        t_fit = time.perf_counter() - t0
        policy = Pol(params, payoffs.max_call, nets)
        cert = certify(S0, params, policy, payoff_fn=payoffs.max_call,
                       device=DEVICE)
        c = cert.diagnostics
        lo, hi = c["ci95"]
        thresh = max(GAP_TARGET, 0.5 * pilot_gap)
        adopt = c["gap"] <= thresh
        print(f"  randomized (own)  : {ev['value']:9.4f}  (stderr {ev['mc_stderr']:.4f})")
        print(f"  L0 (certified)    : {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})")
        print(f"  95% CI            : [{lo:.4f}, {hi:.4f}]   "
              f"{'brackets ref' if lo <= ref <= hi else '*** ref outside ***'} "
              f"({ref:.4f})")
        print(f"  gap               : {c['gap']:.4f}  (pilot {pilot_gap})   "
              f"vs threshold {thresh:.4f}")
        print(f"  fit {t_fit:.0f}s, certify {cert.runtime:.0f}s")
        print(f"  -> {'ADOPT doubled budget for the sweep' if adopt else 'KEEP standard budget'}")
        verdicts.append((name, c["gap"], pilot_gap, adopt))

    print("=" * 72)
    print("SWEEP BUDGET DECISIONS:")
    for name, gap, pilot_gap, adopt in verdicts:
        print(f"  {name:9s}: gap {pilot_gap:.3f} -> {gap:.3f}   "
              f"{'doubled budget' if adopt else 'standard budget'}")
    print(f"\ntotal: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
