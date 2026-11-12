"""Chee against the binomial reference on the single-asset put.

The put of run_dong_check.py at lambda = 0.05 and 0.01, with the truncation
n = 1/lambda. The fraction of stops at zero payoff is printed to check the
guard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src.reference import binomial
from src.methods import chee


def main():
    S0 = 40.0
    base = dict(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                n_iters=3000, batch_size=1024, n_paths=2 ** 18, seed=0)
    ref = binomial.price(S0, Params(**base))
    print(f"binomial reference : {ref}")

    for lam in (0.05, 0.01):
        res = chee.price(S0, Params(**base, lambda_=lam))
        d = res.diagnostics
        print(f"\nchee lambda={lam:g} (n={d['n_truncation']:g}, device={d['device']}): {res}")
        print(f"  error vs reference : {res.value - ref.value:+.4f}")
        print(f"  greedy boundary    : {d['greedy_policy_value']:.4f}")
        print(f"  value net V_0(S0)  : {d['value_net_V0']:.4f}")
        print(f"  MC stderr          : {d['mc_stderr']:.4f}")
        print(f"  -lambda*log(lambda): {d['lambda_bias_scale']:.4f}")
        print(f"  otm_stop_frac      : {d['otm_stop_frac']:.4f}  (guard check: 0)")


if __name__ == "__main__":
    main()
