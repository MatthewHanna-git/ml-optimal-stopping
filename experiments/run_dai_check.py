"""Dai against the binomial reference on the single-asset put.

The put of run_dong_check.py at lambda = 1 (the paper's temperature) and
lambda = 0.1, with the paper's penalty K = 10. The largest stopping
probability is printed to check the penalty does not saturate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src.reference import binomial
from src.methods import dai


def main():
    S0 = 40.0
    base = dict(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                n_iters=3000, batch_size=1024, n_paths=2 ** 18, seed=0)
    ref = binomial.price(S0, Params(**base))
    print(f"binomial reference : {ref}")

    for lam in (1.0, 0.1):
        res = dai.price(S0, Params(**base, lambda_=lam))
        d = res.diagnostics
        print(f"\ndai lambda={lam:g} (device={d['device']}): {res}")
        print(f"  error vs reference : {res.value - ref.value:+.4f}")
        print(f"  greedy boundary    : {d['greedy_policy_value']:.4f}")
        print(f"  value net V_0(S0)  : {d['value_net_V0']:.4f}")
        print(f"  MC stderr          : {d['mc_stderr']:.4f}")
        print(f"  max stop prob      : {d['max_stop_prob']:.2f}  (penalty check: < 1)")
        print(f"  lambda*T*log2 bound: {d['lambda_bias_scale']:.4f}")


if __name__ == "__main__":
    main()
