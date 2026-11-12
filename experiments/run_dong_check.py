"""Dong against the binomial reference on the single-asset put.

The put of Dong (2024, Section 4.2.1): S0 = K = 40, r = 6%, sigma = 40%,
T = 1, 50 exercise dates, at lambda = 1e-3. Prints the randomised policy
value, the greedy policy value and the network's V_0 against the tree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.params import Params
from src.reference import binomial
from src.methods import dong


def main():
    S0 = 40.0
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                    lambda_=1e-3, n_iters=3000, batch_size=1024,
                    n_paths=2 ** 18, seed=0)

    ref = binomial.price(S0, params)
    print(f"binomial reference : {ref}")
    print("training dong (a few minutes on GPU; longer on CPU)...")
    approx = dong.price(S0, params)
    print(f"dong               : {approx}")

    err = approx.value - ref.value
    d = approx.diagnostics
    print(f"\nerror vs reference : {err:+.4f}")
    print(f"MC stderr          : {d['mc_stderr']:.4f}")
    print(f"greedy tau^lambda  : {d['greedy_policy_value']:.4f}")
    print(f"value net V_0(S0)  : {d['value_net_V0']:.4f}")
    print(f"-lambda*log(lambda): {d['lambda_bias_scale']:.4f}")
    print(f"final TD error     : {d['final_td_error']:.2e}")


if __name__ == "__main__":
    main()
