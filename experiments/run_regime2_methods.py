"""The four RL methods on the two-asset max-call at S0 = 100.

Setup as in run_regime2_table.py: K = 100, r = 5%, sigma = 20%,
delta = 10%, T = 3, 100 exercise dates, against the Bermudan product tree at
10 substeps. Set RUN to choose the methods:

  dong      lambda = 1e-2
  dai       lambda = 0.1 at K = 40 and K = 80, to confirm the penalty
            saturates, and lambda = 0.02 at K = 40
  chee      lambda = 0.1 (n = 10) and 0.01 (n = 100), printed beside the
            paper's values at the same n (Table 1)
  dianetti  lambda = 0.01, at its budget
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from src import payoffs
from src.reference import binomial2d
from src.methods import dong, dai, chee, dianetti

# Methods to run: any of "dong", "dai", "chee", "dianetti", or {"all"}
RUN = {"all"}

S0 = 100.0
BASE = dict(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1, dim=2,
            n_steps=100, tree_substeps=10, n_paths=2 ** 18, seed=0)
STD = dict(n_iters=3000, batch_size=1024)
DEEP = dict(n_iters=12000, batch_size=4096)  # Dong and Chee at 100 dates
BIG = dict(n_iters=6000, batch_size=2048, lr=5e-3)     # dianetti budget


def show(tag, res, ref):
    d = res.diagnostics
    extra = "" if d.get("greedy_policy_value") is None \
        else f"  greedy {d['greedy_policy_value']:.4f}"
    td = d.get("final_td_error")
    td_s = f"  TD {td:.2e}" if td is not None else ""
    print(f"{tag:24s} value {res.value:.4f}  err {res.value - ref:+.4f}"
          f"{extra}{td_s}  ({res.runtime:.0f}s)")


def main():
    want = {w.lower() for w in RUN}
    if "all" in want:
        want = {"dong", "dai", "chee", "dianetti"}
    ref = binomial2d.price(S0, Params(**BASE, **STD)).value
    print(f"Bermudan-100 2-D reference: {ref:.4f}\n")

    if "dong" in want:
        show("dong lam=1e-2", dong.price(S0, Params(**BASE, **DEEP, lambda_=1e-2),
                                         payoff_fn=payoffs.max_call), ref)
    if "dai" in want:
        for K in (40.0, 80.0):
            show(f"dai lam=0.1 K={K:.0f}",
                 dai.price(S0, Params(**BASE, **STD, lambda_=0.1, penalty=K),
                           payoff_fn=payoffs.max_call), ref)
        # a smaller lambda separates the temperature bias from the training gap
        show("dai lam=0.02 K=40",
             dai.price(S0, Params(**BASE, **STD, lambda_=0.02, penalty=40.0),
                       payoff_fn=payoffs.max_call), ref)
    if "chee" in want:
        for lam, pia in ((0.1, 13.349), (0.01, 14.213)):
            res = chee.price(S0, Params(**BASE, **DEEP, lambda_=lam),
                             payoff_fn=payoffs.max_call)
            show(f"chee lam={lam:g} (n={1/lam:.0f})", res, ref)
            print(f"{'':24s}  [Chee Table 1 PIA at this n: {pia}]")
    if "dianetti" in want:
        show("dianetti lam=0.01", dianetti.price(S0, Params(**BASE, **BIG, lambda_=0.01),
                                                 payoff_fn=payoffs.max_call), ref)


if __name__ == "__main__":
    main()
