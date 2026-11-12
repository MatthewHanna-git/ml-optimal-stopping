"""Accuracy of the four RL methods on the single-asset put across moneyness.

For each S0 in {36, 40, 44} and each method at lambda = 1e-3: the price,
its error against the binomial reference, the runtime and the final TD
error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src.reference import binomial
from src.methods import REGISTRY

# Dianetti runs at a larger budget (see its docstring); the others use the defaults
OVERRIDES = {"dianetti": dict(n_iters=6000, batch_size=2048, lr=5e-3)}

def method_params(name, base_params):
    if name not in OVERRIDES:
        return base_params
    from src.params import Params as _P
    return _P(**{**base_params.__dict__, **OVERRIDES[name]})

S0_GRID = [36.0, 40.0, 44.0]
LAMBDA = 1e-3


def main():
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                    lambda_=LAMBDA, seed=0)

    header = f"{'S0':>4} | {'reference':>9} | " + " | ".join(
        f"{name+' price':>12} {'error':>8} {'runtime':>8} {'TD':>8}" for name in REGISTRY)
    print(header)
    print("-" * len(header))

    for S0 in S0_GRID:
        ref = binomial.price(S0, params)
        row = f"{S0:4.0f} | {ref.value:9.4f} | "
        cells = []
        for name, method in REGISTRY.items():
            res = method(S0, method_params(name, params))
            td = res.diagnostics.get("final_td_error", float("nan"))
            cells.append(f"{res.value:12.4f} {res.value - ref.value:+8.4f} "
                         f"{res.runtime:7.1f}s {td:8.1e}")
        print(row + " | ".join(cells))


if __name__ == "__main__":
    main()
