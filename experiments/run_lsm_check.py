"""Validation of LSM under the Andersen-Broadie certification.

Each cell fits the LSM policy on N_TRAIN paths and certifies it; L0 is the
policy's value on paths independent of the regression. The in-sample
estimate is printed for comparison.

(a) The single-asset Bermudan put (S0 = K = 40, r = 6%, sigma = 40%, T = 1,
    50 dates), against the binomial value and Longstaff and Schwartz
    (2001, Table 1).
(b) The two-asset max-call of the sweep, S0 in {90, 100, 110}, against the
    tree and Andersen and Broadie (2004, Table 2), whose lower bounds are
    LSM values.
(c) The five-asset max-call, beyond the trees, against the published
    intervals: Andersen and Broadie (2004), Broadie and Cao (2008),
    Longstaff and Schwartz (2001, Section 8.1) and Becker et al. (2021).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.params import Params
from src.reference import binomial, binomial2d, lsm
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_PANEL_A = True
RUN_PANEL_B = True
RUN_PANEL_C = True

N_TRAIN = 2 ** 18            # regression paths (AB used 200k)
N_EVAL = 2 ** 21             # lower-bound paths inside certify (AB used 2M)

# Certification budgets: AB 2004, Table 2 for the 9-date panels; a smaller
# inner budget for the 50-date put, as in run_ab_check.py.
AB_PUT = dict(n_outer=1500, n_inner=4000)
AB_MAXCALL = dict(n_outer=1500, n_inner=10000)

# Published anchors ----------------------------------------------------------
PUB_D2 = {  # AB 2004 Table 2 (their L = LSM policy value) + Becker Table 9
    90.0: dict(ab_L=(8.065, 0.006), ab_U=(8.069, 0.007), ab_ci=(8.053, 8.082),
               binomial=8.075, becker=(8.068, 0.006)),
    100.0: dict(ab_L=(13.907, 0.008), ab_U=(13.915, 0.010), ab_ci=(13.892, 13.934),
                binomial=13.902, becker=(13.901, 0.010)),
    110.0: dict(ab_L=(21.333, 0.009), ab_U=(21.340, 0.010), ab_ci=(21.316, 21.359),
                binomial=21.345, becker=(21.341, 0.008)),
}
PUB_D5 = {  # AB Table 2; Broadie-Cao 2008 CI; LS 2001 §8.1 in-sample; Becker Table 9
    90.0: dict(ab_L=(16.618, 0.008), ab_U=(16.634, 0.010), ab_ci=(16.602, 16.655),
               bc_ci=(16.620, 16.653), ls_in=16.657, becker=(16.631, 0.009)),
    100.0: dict(ab_L=(26.128, 0.010), ab_U=(26.253, 0.020), ab_ci=(26.109, 26.292),
                bc_ci=(26.115, 26.164), ls_in=26.182, becker=(26.147, 0.009)),
    110.0: dict(ab_L=(36.725, 0.011), ab_U=(36.798, 0.017), ab_ci=(36.704, 36.832),
                bc_ci=(36.710, 36.798), ls_in=36.812, becker=(36.774, 0.009)),
}
# ----------------------------------------------------------------------------


def run_cell(S0, params, ref_value=None, pub=None):
    t0 = time.perf_counter()
    policy, v_in, se_in = lsm.fit(S0, params, device=DEVICE, n_train=N_TRAIN)
    t_fit = time.perf_counter() - t0
    n_b = len(next(iter(policy.coeffs.values()))) if policy.coeffs else 0
    print(f"  lsm fit           : {n_b} basis fns, {t_fit:.1f}s"
          + (f", skipped dates {policy.skipped_dates}" if policy.skipped_dates else ""))
    print(f"  lsm in-sample     : {v_in:9.4f}  (stderr {se_in:.4f}; LS-style, "
          f"not a bound)")

    cert = certify(S0, params, policy, device=DEVICE)
    c = cert.diagnostics
    lo, hi = c["ci95"]
    print(f"  L0 (out-of-sample): {c['lower']:9.4f}  (stderr {c['lower_stderr']:.4f})"
          f"   <- the LSM value")
    print(f"  A-B upper U0      : {c['upper']:9.4f}  (stderr {c['delta_stderr']:.4f}"
          f" on delta {c['delta']:.4f})")
    print(f"  95% CI            : [{lo:.4f}, {hi:.4f}]   gap {c['gap']:.4f}   "
          f"({cert.runtime:.1f}s)")
    if ref_value is not None:
        inside = lo <= ref_value <= hi
        print(f"  reference         : {ref_value:9.4f}   "
              f"{'INSIDE CI' if inside else '*** OUTSIDE CI ***'}")
    if pub is not None:
        print(f"  published AB      : L {pub['ab_L'][0]:.3f} ({pub['ab_L'][1]:.3f})"
              f"   U {pub['ab_U'][0]:.3f} ({pub['ab_U'][1]:.3f})"
              f"   CI [{pub['ab_ci'][0]:.3f}, {pub['ab_ci'][1]:.3f}]")
        extra = []
        if "binomial" in pub:
            extra.append(f"binomial {pub['binomial']:.3f}")
        if "bc_ci" in pub:
            extra.append(f"Broadie-Cao CI [{pub['bc_ci'][0]:.3f}, {pub['bc_ci'][1]:.3f}]")
        if "ls_in" in pub:
            extra.append(f"LS in-sample {pub['ls_in']:.3f}")
        extra.append(f"Becker {pub['becker'][0]:.3f} ({pub['becker'][1]:.3f})")
        print("  published (other) : " + "   ".join(extra))


def panel_a():
    print("=" * 72)
    print("(a) 1-D Bermudan put (S0 = K = 40)")
    print("    LS 2001 Table 1, this row: FD American 5.312; LSM 5.308 (0.018)")
    print("=" * 72)
    params = Params(K_strike=40.0, r=0.06, sigma=0.4, T=1.0, n_steps=50,
                    n_paths=N_EVAL, seed=0, **AB_PUT)
    ref = binomial.price(40.0, params)
    print(f"  binomial reference: {ref.value:9.4f}")
    run_cell(40.0, params, ref_value=ref.value)


def panel_b():
    print("=" * 72)
    print("(b) d=2 Bermudan max-call, Becker regime — vs AB's own LSM lower bounds")
    print("=" * 72)
    for S0 in (90.0, 100.0, 110.0):
        params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                        dim=2, n_steps=9, tree_substeps=100, n_paths=N_EVAL,
                        seed=0, **AB_MAXCALL)
        ref = binomial2d.price(S0, params)
        print(f"\nS0 = {S0:.0f}")
        print(f"  binomial2d ref    : {ref.value:9.4f}")
        run_cell(S0, params, ref_value=ref.value, pub=PUB_D2.get(S0))


def panel_c():
    print("=" * 72)
    print("(c) d=5 Bermudan max-call — beyond tree reach; published anchors only")
    print("=" * 72)
    for S0 in (90.0, 100.0, 110.0):
        params = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1,
                        dim=5, n_steps=9, n_paths=N_EVAL, seed=0, **AB_MAXCALL)
        print(f"\nS0 = {S0:.0f}")
        run_cell(S0, params, pub=PUB_D5.get(S0))


def main():
    print(f"device: {DEVICE}")
    t0 = time.perf_counter()
    if RUN_PANEL_A:
        panel_a()
    if RUN_PANEL_B:
        panel_b()
    if RUN_PANEL_C:
        panel_c()
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
