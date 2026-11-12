"""Accuracy of the four RL methods on the two-asset max-call across moneyness.

The max-call of Chee, Frikha and Li (Table 1): K = 100, r = 5%, sigma = 20%,
delta = 10%, T = 3, 100 exercise dates, S0 in {90, 100, 110}. Each method
runs at its settings in CONFIGS, against the Bermudan product tree at 10
substeps. Twelve training runs in all; set RUN to run a subset of methods.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src import payoffs
from src.reference import binomial2d
from src.methods import dong, dai, chee, dianetti

# Methods to run: any subset of CONFIGS, or {"all"}
RUN = {"all"}

S0_GRID = [90.0, 100.0, 110.0]
BASE = dict(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1, dim=2,
            n_steps=100, tree_substeps=10, n_paths=2 ** 18, seed=0)

CONFIGS = {
    "dong":     (dong.price,     dict(lambda_=1e-2, n_iters=12000, batch_size=4096)),
    "dai":      (dai.price,      dict(lambda_=0.1, penalty=40.0,
                                      n_iters=3000, batch_size=1024)),
    "chee":     (chee.price,     dict(lambda_=0.01, n_iters=12000, batch_size=4096)),
    "dianetti": (dianetti.price, dict(lambda_=0.01, n_iters=6000,
                                      batch_size=2048, lr=5e-3)),
}


def main():
    want = {w.lower() for w in RUN}
    if "all" in want:
        want = set(CONFIGS)

    for S0 in S0_GRID:
        ref = binomial2d.price(S0, Params(**BASE)).value
        print(f"\nS0 = {S0:.0f}   reference = {ref:.4f}")
        for name in ("dong", "dai", "chee", "dianetti"):
            if name not in want:
                continue
            fn, cfg = CONFIGS[name]
            res = fn(S0, Params(**BASE, **cfg), payoff_fn=payoffs.max_call)
            d = res.diagnostics
            g = d.get("greedy_policy_value")
            g_s = f"  greedy {g:.4f}" if g is not None else ""
            print(f"  {name:9s} value {res.value:.4f}  err {res.value - ref:+.4f}"
                  f"{g_s}  ({res.runtime:.0f}s)")


if __name__ == "__main__":
    main()
