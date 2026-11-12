# ml-optimal-stopping

Code for the UNSW Master of Financial Mathematics thesis *Machine Learning
for Optimal Stopping*. Eight methods for pricing a Bermudan max-call are
reimplemented in one codebase. Each is reduced to an exercise policy, and
each policy is certified by the Andersen–Broadie interval across
dimensions d = 2 to 50.

The regime is that of Becker, Cheridito and Jentzen (2019, Table 1):
K = 100, r = 5%, δ = 10%, σ = 20%, independent assets, T = 3, nine exercise
dates, S0 ∈ {90, 100, 110}, d ∈ {2, 3, 5, 10, 20, 50}.

## Methods

| Label | Paper | Module |
|---|---|---|
| LSM | Longstaff and Schwartz (2001) | `src/reference/lsm.py` |
| NN-LSM | Lapeyre and Lelong (2021) | `src/methods/nn_lsm.py` |
| RDBDP | Huré, Pham and Warin (2020) | `src/methods/hure.py` |
| DOS | Becker, Cheridito and Jentzen (2019) | `src/methods/becker.py` |
| Dong | Dong (2024) | `src/methods/dong.py` |
| Dai | Dai, Sun, Xu and Zhou (2024) | `src/methods/dai.py` |
| Chee | Chee, Frikha and Li (2026) | `src/methods/chee.py` |
| Dianetti | Dianetti, Ferrari and Xu (2024) | `src/methods/dianetti.py` |

Each module exposes `fit()`, which trains and returns the policy or
networks, and `price(S0, params) -> Result`. The RL methods' networks become
policies through `src/methods/rl_policies.py`. Every policy is certified by
`src/duality/andersen_broadie.py` (`certify()`). Shared pieces:
`src/params.py` (the `Params` bundle), `src/simulate.py` (exact GBM paths),
`src/payoffs.py`, `src/methods/common.py` (the networks) and
`src/reference/binomial*.py` (the 1-D and 2-D trees).

## Setup

Windows, Python 3.12, PyTorch 2.13.0 with CUDA 12.6, one RTX 3080.

    python -m venv .venv
    .venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

Everything also runs on the CPU, slowly. Run every script from the
repository root as `python experiments/<script>.py`. Each script's
docstring states what it runs, why, and any rule fixed before the run.

## Results

The tables and figures in the thesis are built from the CSVs in `results/`.

| Script | Writes | What it is |
|---|---|---|
| `run_sweep.py` | `sweep_results.csv` | The dimension sweep: 8 methods × 6 dimensions × 3 spots, 144 cells (about 9 h, resumable) |
| `run_diag_dong_temperature.py` | `diag_dong_temperature.csv` | Dong and Chee trained at a constant temperature |
| `run_diag_nnlsm_width.py` | `diag_nnlsm_width.csv` | NN-LSM at d = 50 by width and training paths |
| `run_diag_dong_budget.py` | `diag_dong_budget.csv` | Dong at a doubled budget, d = 5, 10, 50 |
| `run_diag_rl_budget.py` | `diag_rl_budget.csv` | Dai, Chee and Dianetti at a doubled budget, d = 20 |
| `run_diag_seed_arm.py` | `diag_seed_arm.csv` | Five extra training seeds for LSM, NN-LSM, RDBDP and DOS at d = 10, 50 |
| `run_diag_seed_arm_rl.py` | `diag_seed_arm_rl.csv` | Five extra training seeds for Dong, Dai, Chee and Dianetti at d = 10, 50 |

The diagnostics change no sweep cell. Trained networks are saved to
`results/nets/` and `results/nets_diag/` (not tracked). Each of these
scripts takes `--smoke` for a quick test run.

## Before the sweep

These print to the console and write no results file.

- **Validation**, against the binomial trees and published intervals:
  `run_ab_check.py` (the certification itself, on tree policies),
  `run_2d_tree_check.py`, `run_lsm_check.py`, `run_becker_check.py`,
  `run_becker_probe.py`, `run_hure_check.py`, `run_nnlsm_check.py` and
  `check_rl_sweep_eval.py` (the RL randomised evaluation).
- **Calibration** of the training budgets: `run_rl_pilot.py`,
  `run_rl_budget_arm.py`, `run_nnlsm_budget_arm.py`, `run_depth_arm.py`.
- **Layer 1** (uncertified): the four RL methods on a 1-D put and a 2-D
  max-call, priced by their own estimators: `run_dong_check.py`,
  `run_dai_check.py`, `run_chee_check.py`, `run_dianetti_check.py`,
  `run_dai_penalty_sweep.py`, `accuracy_table.py`, `convergence_plot.py`,
  `run_regime2_methods.py`, `run_regime2_table.py`.

## Seeds

Sweep cell c = 0, …, 143 (ordered by method, then d, then S0) trains from
s = 10⁶ + 10⁴c. The RL randomised evaluation draws from s + 3, and
certification draws its lower-bound paths from s + 7919 and its outer and
inner paths from s + 7920. No two streams of the sweep coincide
(`run_sweep.py` asserts this at startup). The budget and temperature
diagnostics reuse the seed of the sweep cell they revisit; the seed arms
use their own seed blocks, given in their docstrings.
