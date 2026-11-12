"""Dai's price as the penalty K grows, at fixed lambda = 0.1.

The price should rise towards the reference as the exercise delay ~ r g/K
shrinks, then level off; the penalty is fixed where it does. Once K exceeds
1/dt the per-date stopping probability min(K pi dt, 1) is capped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src.reference import binomial
from src.methods import dai

K_GRID = [10.0, 20.0, 40.0, 80.0]
LAM = 0.1


def main():
    S0 = 40.0
    base = dict(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                lambda_=LAM, n_iters=3000, batch_size=1024,
                n_paths=2 ** 18, seed=0)
    ref = binomial.price(S0, Params(**base))
    print(f"reference: {ref.value:.4f}   (lambda fixed at {LAM})\n")
    print(f"{'K':>5} | {'price':>8} {'error':>8} | {'greedy':>8} {'V0':>8} | {'maxstop':>7}")
    print("-" * 60)
    for K in K_GRID:
        res = dai.price(S0, Params(**base, penalty=K))
        d = res.diagnostics
        print(f"{K:5.0f} | {res.value:8.4f} {res.value - ref.value:+8.4f} | "
              f"{d['greedy_policy_value']:8.4f} {d['value_net_V0']:8.4f} | "
              f"{d['max_stop_prob']:7.2f}")


if __name__ == "__main__":
    main()
