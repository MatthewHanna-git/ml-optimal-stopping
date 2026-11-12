"""Deterministic exercise rules for the four RL methods, for certification.

Each adapter implements exercise(k, S) -> bool, so certify() treats the RL
methods as it does the others. For Dong, Dai and Chee the natural threshold
rules all reduce to the same one:

- Dong: Gibbs intensity >= 1       <=>  exp(-(V - g)/lam) >= 1        <=>  V <= g
- Dai:  Bernoulli prob >= 1/2      <=>  sigmoid(K(g - V)/lam) >= 1/2  <=>  V <= g
- Chee: intensity >= half its max  <=>  mu(alpha, n) >= n/2           <=>  V <= g

so GreedyValuePolicy serves all three, and the methods differ only in how V
was trained.

For Dianetti the fuel is a running minimum, Y_{k+1} = min(Y_k, b_k(X_k)),
so stopping the first time the fuel would fall to a level c is the
memoryless rule b_k(S) <= c. FuelLevelPolicy implements it; c = e^{-1} is
the paper's extraction (Remark 3.10, exact by Theorem 3.9), and
FuelMedianPolicy is c = 1/2, the median of the randomised stopping time.

certify() applies the g > 0 guard to every rule.
"""

import math

import torch

from ..params import Params


class GreedyValuePolicy:
    """Exercise iff V_k(S) <= g(S) (and in the money): dong / dai / chee.

    nets: the per-date ValueNet list from the method's fit(), indexed by
    date l = 0..N-1 (date k uses nets[k]; date N is the maturity rule)."""

    def __init__(self, params: Params, payoff_fn, nets):
        self.K = params.K_strike
        self.n_steps = params.n_steps
        self.payoff_fn = payoff_fn
        self.nets = nets

    @torch.no_grad()
    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        return (self.nets[k](S) <= g) & (g > 0.0)


class FuelLevelPolicy:
    """Exercise iff b_k(S) <= threshold: stop when the remaining fuel (the
    survival probability) would first fall to `threshold`. The default
    e^{-1} is the paper's extraction (Remark 3.10)."""

    def __init__(self, params: Params, payoff_fn, nets,
                 threshold: float = math.exp(-1.0)):
        self.K = params.K_strike
        self.n_steps = params.n_steps
        self.payoff_fn = payoff_fn
        self.nets = nets                        # BoundaryNet list
        self.threshold = float(threshold)

    @torch.no_grad()
    def exercise(self, k: int, S: torch.Tensor) -> torch.Tensor:
        g = self.payoff_fn(S, self.K)
        if k >= self.n_steps:
            return g > 0.0
        return (self.nets[k](S) <= self.threshold) & (g > 0.0)


class FuelMedianPolicy(FuelLevelPolicy):
    """The median rule b_k(S) <= 1/2."""

    def __init__(self, params: Params, payoff_fn, nets):
        super().__init__(params, payoff_fn, nets, threshold=0.5)
