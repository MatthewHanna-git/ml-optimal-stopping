"""Diagnostic: LSM on a generic basis, without the structure of its own.

The sweep's LSM regresses on a basis built for the symmetric max-call: the
prices sorted, powers of the largest, neighbouring products. The networks of
the sweep receive the prices in a fixed asset order (and the payoff) and must
find that structure. run_diag_sorted_inputs.py handed the structure to the
networks; this run takes it from LSM, completing the two-by-two

                    handed the structure      must find it
    polynomial      LSM (the sweep)           LSM, generic basis (this run)
    network         sorted NN-LSM, DOS        NN-LSM, DOS (the sweep)

Each cell is the sweep's LSM cell (same 2^18 training paths, seed and
certification streams) with one change, the basis:

    poly{p}    all monomials of total degree <= p in y = S/K, unsorted
               (the complete family of Lapeyre and Lelong, 2021)
    poly{p}g   the same in (y, g/K): the inputs the networks receive

p = 2 at every d; p = 3 for d <= MAX_D_DEG3 (beyond it the family outgrows
the training paths and memory). The regression is the sweep's, in-the-money
paths only, solved by least squares through chunked normal equations in
double precision.

Each cell is compared with the sweep's LSM on the same paths, and a
difference is read only where it exceeds 2 combined standard errors.

Writes results/diag_generic_lsm.csv (appended per cell, resumable). About an
hour on an RTX 3080.

Usage:  python experiments/run_diag_generic_lsm.py
        python experiments/run_diag_generic_lsm.py --smoke
"""

import csv
import itertools
import math
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch

import run_sweep as sweep
from src import payoffs
from src.simulate import gbm_paths
from src.duality import certify

# ----------------------------- RUN settings ---------------------------------
DEVICE = sweep.DEVICE
SMOKE = sweep.SMOKE
RESULTS = sweep.RESULTS
CSV_PATH = RESULTS / ("smoke_diag_generic_lsm.csv" if SMOKE else "diag_generic_lsm.csv")
SWEEP_CSV = RESULTS / "sweep_results.csv"

DIMS = list(sweep.DIMS)                  # [2, 3, 5, 10, 20, 50]; smoke: [2, 5]
S0_LIST = sweep.S0_LIST
BASES = [("poly2g", 2, True), ("poly2", 2, False),
         ("poly3g", 3, True), ("poly3", 3, False)]
MAX_D_DEG3 = 10
ROWS = 2 ** 15                           # rows per chunk, in fit and in exercise
READ_Z = 2.0

FIELDS = ["basis", "d", "S0", "n_basis", "seed", "n_train",
          "L0", "L0_stderr", "delta", "delta_stderr", "U0", "gap", "ci_lo", "ci_hi",
          "in_sample", "skipped_dates",
          "lsm_L0", "lsm_L0_stderr", "lsm_gap", "dL0", "dL0_se", "z", "read",
          "fit_s", "cert_s", "timestamp"]
# ----------------------------------------------------------------------------


class GenericBasis:
    """All monomials of total degree <= p in v = (S/K) or (S/K, g/K), unsorted."""

    def __init__(self, dim, degree, with_payoff, K):
        self.K, self.with_payoff = K, with_payoff
        n_var = dim + (1 if with_payoff else 0)
        self.terms = [c for q in range(degree + 1)
                      for c in itertools.combinations_with_replacement(range(n_var), q)]
        # each term as `degree` indices into (v, 1), padded with the ones column
        self._idx = torch.tensor([list(c) + [n_var] * (degree - len(c)) for c in self.terms])

    @property
    def size(self):
        return len(self.terms)

    def __call__(self, S):
        v = S / self.K
        if self.with_payoff:
            v = torch.cat([v, (payoffs.max_call(S, self.K) / self.K).unsqueeze(-1)], dim=-1)
        v1 = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
        idx = self._idx.to(S.device)
        return torch.prod(v1[:, idx], dim=-1)            # (rows, B)


class GenericLSMPolicy:
    """Exercise iff in the money and g >= fitted continuation; evaluated in chunks."""

    def __init__(self, params, basis, coeffs):
        self.K, self.n_steps = params.K_strike, params.n_steps
        self.basis, self.coeffs = basis, coeffs
        self.skipped_dates = [k for k in range(1, params.n_steps) if k not in coeffs]
        self._cache = {}

    def _beta(self, k, device):
        if (k, device) not in self._cache:
            self._cache[(k, device)] = self.coeffs[k].to(device)
        return self._cache[(k, device)]

    @torch.no_grad()
    def exercise(self, k, S):
        g = payoffs.max_call(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        if k not in self.coeffs:
            return torch.zeros_like(g, dtype=torch.bool)
        beta = self._beta(k, S.device)
        C = torch.cat([self.basis(S[i:i + ROWS]) @ beta for i in range(0, S.shape[0], ROWS)])
        return (g > 0.0) & (g >= C)


@torch.no_grad()
def fit(S0, params, basis, n_train):
    """The sweep LSM's recursion (lsm.fit) with the generic basis."""
    N, dt = params.n_steps, params.T / params.n_steps
    gen = torch.Generator(device=DEVICE).manual_seed(params.seed)
    paths = gbm_paths(S0, params, n_paths=n_train, device=DEVICE, generator=gen)
    B = basis.size
    disc_step = math.exp(-params.r * dt)
    cf = payoffs.max_call(paths[:, N, :], params.K_strike).clone()
    coeffs = {}
    for k in range(N - 1, 0, -1):
        cf *= disc_step
        S_k = paths[:, k, :]
        g_k = payoffs.max_call(S_k, params.K_strike)
        itm = torch.nonzero(g_k > 0.0, as_tuple=True)[0]
        if itm.numel() < max(2 * B, 50):
            continue
        XtX = torch.zeros(B, B, dtype=torch.float64, device=DEVICE)
        Xty = torch.zeros(B, dtype=torch.float64, device=DEVICE)
        for i in range(0, itm.numel(), ROWS):
            idx = itm[i:i + ROWS]
            X = basis(S_k[idx]).double()
            XtX += X.T @ X
            Xty += X.T @ cf[idx].double()
        # B x B system solved on the CPU by SVD (gelsd), robust to near-collinearity
        beta = torch.linalg.lstsq(XtX.cpu(), Xty.cpu().unsqueeze(-1),
                                  driver="gelsd").solution.squeeze(-1).to(DEVICE)
        coeffs[k] = beta.float().cpu()
        for i in range(0, itm.numel(), ROWS):
            idx = itm[i:i + ROWS]
            C = (basis(S_k[idx]).double() @ beta).float()
            ex = g_k[idx] >= C
            cf[idx[ex]] = g_k[idx[ex]]
    in_sample = (cf * disc_step).mean().item()
    return GenericLSMPolicy(params, basis, coeffs), in_sample


def load_sweep_rows():
    rows = {}
    with open(SWEEP_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["method"], int(r["d"]), float(r["S0"]))] = r
    return rows


def load_done():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as f:
        return {(r["basis"], int(r["d"]), float(r["S0"])) for r in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def main():
    sweep.env_banner()
    print(f"diag csv: {CSV_PATH}")
    print(f"RULE: read dL0 against the sweep's LSM only where |dL0| > {READ_Z:g} x combined s.e.")
    srows = load_sweep_rows()
    done = load_done()
    if done:
        print(f"resuming: {len(done)} cells already recorded, will be skipped")
    t_all = time.perf_counter()

    for name, degree, with_g in BASES:
        for d in DIMS:
            if degree >= 3 and d > MAX_D_DEG3:
                continue
            for S0 in S0_LIST:
                if (name, d, S0) in done:
                    print(f"skip (done): {name} d={d} S0={S0:.0f}")
                    continue
                seed = sweep.cell_seed("lsm", d, S0)          # the sweep LSM cell's seed
                params = sweep.make_params(d, seed, {})
                basis = GenericBasis(d, degree, with_g, params.K_strike)
                sweep._sync(); t0 = time.perf_counter()
                policy, v_in = fit(S0, params, basis, sweep.N_TRAIN)
                sweep._sync(); fit_s = time.perf_counter() - t0
                cert = certify(S0, params, policy, payoff_fn=payoffs.max_call, device=DEVICE)
                c = cert.diagnostics

                base = None if SMOKE else srows.get(("lsm", d, S0))
                b_L0 = float(base["L0"]) if base else float("nan")
                b_se = float(base["L0_stderr"]) if base else float("nan")
                b_gap = float(base["gap"]) if base else float("nan")
                dL0 = c["lower"] - b_L0
                dL0_se = math.sqrt(c["lower_stderr"] ** 2 + b_se ** 2)
                z = dL0 / dL0_se if dL0_se > 0 else float("nan")
                read = abs(z) > READ_Z

                print(f"{name:6s} d={d:<3d} S0={S0:<4.0f} B={basis.size:<5d} "
                      f"L0={c['lower']:9.4f} ({c['lower_stderr']:.4f})  gap={c['gap']:.4f}  "
                      f"| sweep LSM L0={b_L0:9.4f} gap={b_gap:.4f}  dL0={dL0:+.4f} "
                      f"(z={z:+.1f}{', read' if read else ''})"
                      f"{'  skipped dates ' + str(policy.skipped_dates) if policy.skipped_dates else ''}"
                      f"\n{'':7s}fit {fit_s:.0f}s  cert {cert.runtime:.0f}s", flush=True)

                append_row({
                    "basis": name, "d": d, "S0": S0, "n_basis": basis.size,
                    "seed": seed, "n_train": sweep.N_TRAIN,
                    "L0": c["lower"], "L0_stderr": c["lower_stderr"],
                    "delta": c["delta"], "delta_stderr": c["delta_stderr"],
                    "U0": c["upper"], "gap": c["gap"],
                    "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1],
                    "in_sample": v_in, "skipped_dates": " ".join(map(str, policy.skipped_dates)),
                    "lsm_L0": b_L0, "lsm_L0_stderr": b_se, "lsm_gap": b_gap,
                    "dL0": dL0, "dL0_se": dL0_se, "z": z, "read": read,
                    "fit_s": round(fit_s, 1), "cert_s": round(cert.runtime, 1),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })

    print(f"\ntotal {time.perf_counter() - t_all:.0f}s\nresults: {CSV_PATH}")


if __name__ == "__main__":
    main()
