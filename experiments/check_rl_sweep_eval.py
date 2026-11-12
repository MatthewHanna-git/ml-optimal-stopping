"""Tests of rl_sweep_eval.randomised_value and FuelLevelPolicy (CPU, about a
minute).

1. Where the two conventions coincide (payoff always positive, no stop at
   date 0), randomised_value agrees with each method's own evaluate():
   within Monte Carlo noise for the sampled methods, exactly for Dianetti.
2. With the real payoff and a network that wants to stop at g = 0,
   randomised_value never stops there.
3. FuelLevelPolicy at e^{-1} and at 1/2 differ when b lies between them.
"""
import math, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.params import Params
from src import payoffs
from src.methods import dong, dai, chee, dianetti
from src.methods.rl_sweep_eval import randomised_value
from src.methods.rl_policies import FuelLevelPolicy, FuelMedianPolicy

torch.manual_seed(0)
dev = "cpu"
P = Params(K_strike=100.0, r=0.05, sigma=0.2, T=3.0, delta=0.1, dim=2, n_steps=9,
           n_paths=2 ** 16, seed=11, lambda_=1e-2, penalty=40.0, n_iters=30,
           batch_size=256, lr=1e-3)

def shifted(S, K):                 # always-positive payoff: guard never binds
    return payoffs.max_call(S, K) + 50.0

class Never(torch.nn.Module):      # V = +inf -> intensity 0 (rate branch)
    def forward(self, S): return torch.full((S.shape[0],), 1e9)
class FullFuel(torch.nn.Module):   # b = 1 -> no spend (Dianetti)
    def forward(self, S): return torch.ones(S.shape[0])

print("1. equivalence with evaluate() when conventions coincide")
for name, mod in [("dong", dong), ("dai", dai), ("chee", chee), ("dianetti", dianetti)]:
    p = P if name != "dianetti" else Params(**{**P.__dict__, "lr": 5e-3})
    nets, _ = mod.fit(100.0, p, payoff_fn=shifted, device=dev)
    nets[0] = Never() if name != "dianetti" else FullFuel()
    torch.manual_seed(p.seed + 3)              # CPU: global stream == Generator(seed+3)
    old = mod.evaluate(nets, 100.0, p, payoff_fn=shifted, device=dev)
    new = randomised_value(name, nets, 100.0, p, shifted, dev, seed=p.seed + 3)
    se = math.hypot(old["mc_stderr"], new["mc_stderr"])
    z = (new["value"] - old["value"]) / se
    tag = "exact" if name == "dianetti" else f"{z:+.2f} se"
    ok = abs(new["value"] - old["value"]) < 1e-4 if name == "dianetti" else abs(z) < 3
    print(f"  {name:9s} old={old['value']:.4f} new={new['value']:.4f}  ({tag})  "
          f"{'OK' if ok else 'FAIL'}")
    if name != "dianetti":
        og = old["greedy_policy_value"]; ng = new["greedy_value"]
        print(f"  {'':9s} greedy old={og:.4f} new={ng:.4f} (same-path diag; MC noise only)")
    assert ok

print("2. guard: new evaluation never stops at g = 0")
class Eager(torch.nn.Module):      # V = -inf -> wants to stop everywhere
    def forward(self, S): return torch.full((S.shape[0],), -1e9)
class Dump(torch.nn.Module):       # b = 0 -> dump all fuel at the first date
    def forward(self, S): return torch.zeros(S.shape[0])
for name, Net in [("dong", Eager), ("dai", Eager), ("chee", Eager), ("dianetti", Dump)]:
    nets = [Net() for _ in range(P.n_steps)]
    new = randomised_value(name, nets, 100.0, P, payoffs.max_call, dev, seed=5)
    # under 'stop at the first date with g>0' the value must equal that rule's
    # value computed independently:
    g = torch.Generator().manual_seed(5)
    from src.simulate import gbm_paths
    S = gbm_paths(100.0, P, n_paths=P.n_paths, device=dev, generator=g)
    dt = P.T / P.n_steps
    pay = torch.zeros(P.n_paths); alive = torch.ones(P.n_paths, dtype=torch.bool)
    for k in range(1, P.n_steps + 1):
        gk = payoffs.max_call(S[:, k, :], P.K_strike)
        stop = alive & ((gk > 0) | (k == P.n_steps))
        pay[stop] = math.exp(-P.r * k * dt) * gk[stop]; alive &= ~stop
    ref = pay.mean().item()
    print(f"  {name:9s} value={new['value']:.4f} first-ITM-date rule={ref:.4f} "
          f"otm_stop_frac={new['otm_stop_frac']}  "
          f"{'OK' if abs(new['value'] - ref) < 1e-4 and new['otm_stop_frac'] == 0 else 'FAIL'}")
    assert abs(new["value"] - ref) < 1e-4 and new["otm_stop_frac"] == 0
    if name == "dianetti":
        torch.manual_seed(5)
        old = dianetti.evaluate(nets, 100.0, P, payoff_fn=payoffs.max_call, device=dev)
        print(f"  {'':9s} old evaluate(): value={old['value']:.4f} otm_spend_frac={old['otm_spend_frac']:.3f} (spends fuel at g=0 / date 0)")

print("3. FuelLevelPolicy thresholds")
class Const(torch.nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, S): return torch.full((S.shape[0],), self.c)
S = torch.tensor([[120.0, 90.0]])
for c in (0.3, 0.4, 0.6):
    nets = [Const(c)] * 9
    e = FuelLevelPolicy(P, payoffs.max_call, nets).exercise(3, S).item()
    h = FuelLevelPolicy(P, payoffs.max_call, nets, threshold=0.5).exercise(3, S).item()
    m = FuelMedianPolicy(P, payoffs.max_call, nets).exercise(3, S).item()
    print(f"  b={c}: e^-1 rule {e}, 1/2 rule {h}, FuelMedianPolicy {m}")
    assert e == (c <= math.exp(-1)) and h == (c <= 0.5) and m == h
S0 = torch.tensor([[95.0, 90.0]])   # g = 0: guard
assert not FuelLevelPolicy(P, payoffs.max_call, [Const(0.1)] * 9).exercise(3, S0).item()
print("  guard at g=0: OK")
print("ALL OK")
