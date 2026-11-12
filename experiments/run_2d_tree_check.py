"""The two-asset binomial tree against Chee, Frikha and Li (Table 1).

(a) The 100-step tree with exercise at every step, the construction of the
    published binomial column, against its values.
(b) The Bermudan reference used for the RL methods, exercise on the 100
    dates only, at 5 and 10 substeps.
(c) The American limit, a 1000-step tree, for context.

Setup: K = 100, r = 5%, sigma = 20%, delta = 10%, T = 3, the max-call on two
independent assets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src.reference import binomial2d

PUB = {90.0: 8.296, 100.0: 14.211, 110.0: 21.799}
BASE = dict(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1, dim=2)


def main():
    print("(a) Chee Table 1 binomial column (their construction: 100-step tree)")
    p = Params(**BASE, n_steps=100, tree_substeps=1)
    for S0, pub in PUB.items():
        r = binomial2d.price(S0, p)
        print(f"    S0={S0:5.0f}: ours {r.value:7.3f}   published {pub:7.3f}   "
              f"diff {r.value - pub:+.3f}")

    print("\n(b) Bermudan reference (100 dates, substep convergence)")
    for sub in (5, 10):
        p = Params(**BASE, n_steps=100, tree_substeps=sub)
        vals = [f"S0={S0:.0f}: {binomial2d.price(S0, p).value:.4f}" for S0 in PUB]
        print(f"    sub={sub:2d}:  " + "   ".join(vals))

    print("\n(c) Converged American limit (context)")
    p = Params(**BASE, n_steps=1000, tree_substeps=1)
    for S0 in PUB:
        r = binomial2d.price(S0, p)
        print(f"    S0={S0:5.0f}: {r.value:.4f}   ({r.runtime:.0f}s)")


if __name__ == "__main__":
    main()
