"""Validation of the Andersen-Broadie certification on tree policies.

The policies certified here are the near-optimal exercise rules implied by
the binomial trees, so any discrepancy belongs to the certification, not to
a learned method.

(a) The single-asset Bermudan put (S0 = K = 40, r = 6%, sigma = 40%, T = 1,
    50 dates): the 95% interval should contain the binomial value.
(b) The two-asset Bermudan max-call of the sweep (K = 100, r = 5%,
    delta = 10%, sigma = 20%, T = 3, 9 dates), S0 in {90, 100, 110}: the
    interval should contain the two-asset tree value and agree with the
    published intervals (Andersen and Broadie 2004, Table 2; Becker et al.
    2021, Table 9).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import binomial, binomial2d
from src.duality import certify, PutTreePolicy1D, MaxCallTreePolicy2D

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_PANEL_A = True
RUN_PANEL_B = True

# Panel (a): the put. Its 50 dates make the nested simulation about 25 times
# the work of panel (b), so the inner budget is smaller; a smaller inner
# budget only widens the interval.
A = dict(S0=40.0, n_paths=2 ** 20, n_outer=1500, n_inner=4000)

# Panel (b): the two-asset max-call, at the budgets of AB 2004, Table 2.
B = dict(S0_list=(90.0, 100.0, 110.0), n_paths=2 ** 21, n_outer=1500,
         n_inner=10000, tree_substeps=100)

# Published anchors for panel (b), n = 2 assets (AB 2004 Table 2; identical
# setup in Becker et al. 2021 Table 9):
PUB = {
    90.0: dict(lower=(8.065, 0.006), upper=(8.069, 0.007),
               ci=(8.053, 8.082), binomial=8.075, becker=(8.068, 0.006)),
    100.0: dict(lower=(13.907, 0.008), upper=(13.915, 0.010),
                ci=(13.892, 13.934), binomial=13.902, becker=(13.901, 0.010)),
    110.0: dict(lower=(21.333, 0.009), upper=(21.340, 0.010),
                ci=(21.316, 21.359), binomial=21.345, becker=(21.341, 0.008)),
}
# ----------------------------------------------------------------------------


def report(tag, res, ref_value, pub=None):
    d = res.diagnostics
    lo, hi = d["ci95"]
    inside = lo <= ref_value <= hi
    print(f"  lower L0          : {d['lower']:9.4f}  (stderr {d['lower_stderr']:.4f})")
    print(f"  upper U0          : {d['upper']:9.4f}  (stderr {d['delta_stderr']:.4f}"
          f" on delta {d['delta']:.4f})")
    print(f"  95% CI            : [{lo:.4f}, {hi:.4f}]   gap {d['gap']:.4f}")
    print(f"  point (L+delta/2) : {res.value:9.4f}")
    print(f"  reference         : {ref_value:9.4f}   "
          f"{'INSIDE CI' if inside else '*** OUTSIDE CI ***'}")
    if pub is not None:
        print(f"  published (AB T2) : L {pub['lower'][0]:.3f} ({pub['lower'][1]:.3f})"
              f"   U {pub['upper'][0]:.3f} ({pub['upper'][1]:.3f})"
              f"   CI [{pub['ci'][0]:.3f}, {pub['ci'][1]:.3f}]"
              f"   binomial {pub['binomial']:.3f}")
        print(f"  published (Becker): {pub['becker'][0]:.3f} ({pub['becker'][1]:.3f})")
    print(f"  t=0 payoff vs L0  : {d['t0_payoff']:.4f} vs {d['lower']:.4f}"
          f"   (t=0 exercise excluded; must not dominate)")
    print(f"  runtime           : {res.runtime:.1f}s   [{tag}]")


def panel_a():
    print("=" * 72)
    print("(a) 1-D Bermudan put (S0 = K = 40) — tree-implied threshold policy")
    print("=" * 72)
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                    n_paths=A["n_paths"], n_outer=A["n_outer"],
                    n_inner=A["n_inner"], seed=0)
    ref = binomial.price(A["S0"], params)
    print(f"  binomial reference: {ref.value:9.4f}   ({ref.runtime:.1f}s)")
    policy = PutTreePolicy1D(A["S0"], params)
    drift = policy.tree_value - ref.value
    print(f"  policy tree value : {policy.tree_value:9.4f}   "
          f"(vs reference {drift:+.2e}; must be ~0 — same tree)")
    res = certify(A["S0"], params, policy, device=DEVICE)
    report("panel a", res, ref.value)


def panel_b():
    print("=" * 72)
    print("(b) d=2 Bermudan max-call, BECKER REGIME (K=100, r=5%, delta=10%,")
    print("    sigma=20%, T=3, 9 dates) — tree-implied continuation policy")
    print("=" * 72)
    for S0 in B["S0_list"]:
        params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                        dim=2, n_steps=9, tree_substeps=B["tree_substeps"],
                        n_paths=B["n_paths"], n_outer=B["n_outer"],
                        n_inner=B["n_inner"], seed=0)
        ref = binomial2d.price(S0, params)
        print(f"\nS0 = {S0:.0f}")
        print(f"  binomial2d ref    : {ref.value:9.4f}   ({ref.runtime:.1f}s, "
              f"Bermudan-9, sub={B['tree_substeps']})")
        policy = MaxCallTreePolicy2D(S0, params)
        drift = policy.tree_value - ref.value
        print(f"  policy tree value : {policy.tree_value:9.4f}   "
              f"(vs reference {drift:+.2e}; must be ~0 — same tree)")
        res = certify(S0, params, policy, device=DEVICE)
        report(f"panel b S0={S0:.0f}", res, ref.value, pub=PUB.get(S0))


def main():
    print(f"device: {DEVICE}")
    t0 = time.perf_counter()
    if RUN_PANEL_A:
        panel_a()
    if RUN_PANEL_B:
        panel_b()
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
