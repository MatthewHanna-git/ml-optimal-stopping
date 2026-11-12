"""The four RL methods on the two-asset max-call of the sweep, certified.

Setup: K = 100, r = 5%, delta = 10%, sigma = 20%, T = 3, 9 exercise dates,
S0 in {90, 100, 110}, with

    dong      lambda = 1e-2
    dai       lambda = 0.1, K = 40 (and K = 80 at S0 = 100, to confirm the
              penalty saturates)
    chee      lambda = 1e-2 (n = 100)
    dianetti  lambda = 1e-2, at its budget

Each cell trains once and reports three values: the randomised policy, the
greedy policy on the same paths, and the certified interval of the
deterministic rule against the two-asset tree. L0 less the randomised value
is the smoothing cost.

Rules applied to the results:
  R1. A method passes if its interval contains the tree value at all three
      S0 and its gap at S0 = 100 is at most 0.15.
  R2. Dai's penalty saturates if |L0(K = 40) - L0(K = 80)| is within two
      combined standard errors at S0 = 100.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src import payoffs
from src.reference import binomial2d
from src.methods import dong, dai, chee, dianetti
from src.methods.rl_policies import GreedyValuePolicy, FuelMedianPolicy
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN = {"dong", "dai", "chee", "dianetti"}   # methods to run
S0_LIST = (90.0, 100.0, 110.0)

N_EVAL = 2 ** 21
AB = dict(n_outer=1500, n_inner=10000)
STD = dict(n_iters=3000, batch_size=1024)                 # standard budget
BIG = dict(n_iters=6000, batch_size=2048, lr=5e-3)        # Dianetti's budget

GAP_TOL = 0.15                              # R1 gap rule at S0=100

METHODS = {
    "dong": dict(mod=dong, pol=GreedyValuePolicy, budget=STD,
                 pd=dict(lambda_=1e-2)),
    "dai": dict(mod=dai, pol=GreedyValuePolicy, budget=STD,
                pd=dict(lambda_=0.1, penalty=40.0)),
    "chee": dict(mod=chee, pol=GreedyValuePolicy, budget=STD,
                 pd=dict(lambda_=1e-2)),
    "dianetti": dict(mod=dianetti, pol=FuelMedianPolicy, budget=BIG,
                     pd=dict(lambda_=1e-2)),
}
# ----------------------------------------------------------------------------


def make_params(pd, budget, seed):
    return Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                  dim=2, n_steps=9, tree_substeps=100, n_paths=N_EVAL,
                  seed=seed, **pd, **budget, **AB)


_SEED_BASE = {"dong": 21000, "dai": 22000, "chee": 23000, "dianetti": 24000}


def run_cell(name, cfg, S0, ref, extra_tag=""):
    params = make_params(cfg["pd"], cfg["budget"],
                         seed=_SEED_BASE[name] + int(S0)
                         + (7 if "K=80" in extra_tag else 0))
    t0 = time.perf_counter()
    nets, tdiag = cfg["mod"].fit(S0, params, payoff_fn=payoffs.max_call,
                                 device=DEVICE)
    ev = cfg["mod"].evaluate(nets, S0, params, payoff_fn=payoffs.max_call,
                             device=DEVICE)
    t_fit = time.perf_counter() - t0
    policy = cfg["pol"](params, payoffs.max_call, nets)
    cert = certify(S0, params, policy, payoff_fn=payoffs.max_call,
                   device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    inside = lo <= ref <= hi
    smoothing = c["lower"] - ev["value"]

    print(f"\n{name}{extra_tag}, S0 = {S0:.0f}   (fit+eval {t_fit:.0f}s, "
          f"certify {cert.runtime:.0f}s)")
    print(f"  randomized (own)  : {ev['value']:9.4f}  (stderr {ev['mc_stderr']:.4f})")
    if ev.get("greedy_policy_value") is not None:
        print(f"  greedy (own eval) : {ev['greedy_policy_value']:9.4f}")
    print(f"  L0 (certified)    : {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})")
    print(f"  A-B upper U0      : {c['upper']:9.4f}  (delta {c['delta']:.4f} "
          f"+- {c['delta_stderr']:.4f})")
    print(f"  95% CI            : [{lo:.4f}, {hi:.4f}]   gap {c['gap']:.4f}")
    print(f"  reference         : {ref:9.4f}   "
          f"{'INSIDE CI' if inside else '*** OUTSIDE CI ***'}")
    print(f"  smoothing cost    : {smoothing:+9.4f}  (certified greedy minus "
          f"randomized)")
    return dict(inside=inside, gap=c["gap"], L0=c["lower"],
                se=c["lower_stderr"])


def main():
    print(f"device: {DEVICE}")
    t_all = time.perf_counter()

    refs = {}
    ref_params = make_params(dict(lambda_=1e-2), STD, seed=0)
    for S0 in S0_LIST:
        refs[S0] = binomial2d.price(S0, ref_params).value
    print("binomial2d refs (Bermudan-9, sub=100): "
          + "   ".join(f"S0={s:.0f}: {v:.4f}" for s, v in refs.items()))

    verdicts = {}
    for name, cfg in METHODS.items():
        if name not in RUN:
            continue
        print("=" * 72)
        print(f"{name}  —  settings {cfg['pd']}, budget {cfg['budget']}")
        print("=" * 72)
        cells = [run_cell(name, cfg, S0, refs[S0]) for S0 in S0_LIST]
        gap_at_100 = (cells[S0_LIST.index(100.0)]["gap"]
                      if 100.0 in S0_LIST else max(x["gap"] for x in cells))
        ok = all(x["inside"] for x in cells) and gap_at_100 <= GAP_TOL
        verdicts[name] = ok

    if "dai" in RUN:
        print("=" * 72)
        print("dai K=80 saturation arm (S0=100)  —  rule R2")
        print("=" * 72)
        cfg80 = dict(METHODS["dai"], pd=dict(lambda_=0.1, penalty=80.0))
        x40 = run_cell("dai", METHODS["dai"], 100.0, refs[100.0], " [K=40]")
        x80 = run_cell("dai", cfg80, 100.0, refs[100.0], " [K=80]")
        d = abs(x40["L0"] - x80["L0"])
        tol = 2 * (x40["se"] ** 2 + x80["se"] ** 2) ** 0.5
        print(f"\n  |L0(40) - L0(80)| = {d:.4f}  vs 2*stderr = {tol:.4f}   "
              f"{'SATURATION HOLDS' if d <= tol else '*** RE-SWEEP K ***'}")

    print("=" * 72)
    print("R1 (all CIs contain the tree value, gap(S0=100) <= "
          f"{GAP_TOL}):")
    for name, ok in verdicts.items():
        print(f"  {name:9s}: {'PASSES R1' if ok else 'FAILS R1'}")
    print(f"\ntotal: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
