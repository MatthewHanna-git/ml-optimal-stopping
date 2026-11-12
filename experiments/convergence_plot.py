"""Convergence of the four RL methods in the temperature, on the at-the-money
put.

Plots each method's absolute error against the binomial reference over a
grid of lambda, on the common axis -lambda log lambda, with the Monte Carlo
standard error as error bars and the line y = x for reference. Writes
results/convergence.png.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

S0 = 40.0
LAMBDAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 1e-3]


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    base = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50, seed=0)
    ref = binomial.price(S0, base)
    print(f"reference: {ref.value:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    xs_line = sorted(-l * math.log(l) for l in LAMBDAS)
    ax.plot(xs_line, xs_line, "k--", lw=1, label=r"$-\lambda\log\lambda$ rate")

    for name, method in REGISTRY.items():
        xs, errs, stderrs = [], [], []
        for lam in LAMBDAS:
            p = Params(**{**base.__dict__, "lambda_": lam})
            res = method(S0, method_params(name, p))
            xs.append(-lam * math.log(lam))
            errs.append(abs(res.value - ref.value))
            stderrs.append(res.diagnostics.get("mc_stderr", 0.0))
            print(f"{name}  lambda={lam:<7g} price={res.value:.4f} "
                  f"|err|={errs[-1]:.4f} ({res.runtime:.0f}s)")
        ax.errorbar(xs, errs, yerr=stderrs, marker="o", capsize=3, label=name)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$-\lambda \log \lambda$")
    ax.set_ylabel("|price error| vs binomial reference")
    ax.set_title(f"Convergence in the temperature, S0={S0:.0f} ATM put")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "convergence.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
