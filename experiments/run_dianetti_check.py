"""Dianetti against the binomial reference on the single-asset put.

The put of run_dong_check.py at lambda = 0.05 and 0.01, at Dianetti's
budget (6000 iterations of 2048 paths, lr 5e-3).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src.reference import binomial
from src.methods import dianetti


def main():
    S0 = 40.0
    base = dict(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                n_iters=6000, batch_size=2048, lr=5e-3,
                n_paths=2 ** 18, seed=0)
    ref = binomial.price(S0, Params(**base))
    print(f"binomial reference : {ref}")

    for lam in (0.05, 0.01):
        res = dianetti.price(S0, Params(**base, lambda_=lam))
        d = res.diagnostics
        print(f"\ndianetti lambda={lam:g} (device={d['device']}): {res}")
        print(f"  error vs reference : {res.value - ref.value:+.4f}")
        print(f"  regularised J      : {d['regularised_value_J']:.4f}")
        print(f"  MC stderr          : {d['mc_stderr']:.4f}")
        print(f"  terminal fuel      : {d['mean_terminal_fuel']:.3f}")
        print(f"  otm spend frac     : {d['otm_spend_frac']:.4f}")
        print(f"  lambda*T/e scale   : {d['lambda_bias_scale']:.4f}")


if __name__ == "__main__":
    main()
