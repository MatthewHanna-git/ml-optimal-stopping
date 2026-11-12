"""The networks shared by the learned methods.

Every method uses the same trunk: Params.arch hidden layers (2 x 50 by
default), one network per exercise date, with the payoff as an input and
prices normalised by the strike. Only the head differs, by what each method
learns: a value, a fuel boundary, a stopping logit or a gradient.
"""

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """V(S) = g(S) + K * NN(S/K, g/K): one instance per timestep."""

    def __init__(self, dim, hidden, K_strike, payoff_fn):
        super().__init__()
        self.K = K_strike
        self.payoff_fn = payoff_fn
        layers, width_in = [], dim + 1          # inputs: S (d coords) and g
        for h in hidden:
            layers += [nn.Linear(width_in, h), nn.ReLU()]
            width_in = h
        layers += [nn.Linear(width_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, S):                        # S: (batch, d)
        g = self.payoff_fn(S, self.K)            # (batch,)
        x = torch.cat([S / self.K, (g / self.K).unsqueeze(-1)], dim=-1)
        return g + self.K * self.net(x).squeeze(-1)


class BoundaryNet(nn.Module):
    """Fuel boundary b(S) in [0, 1] for Dianetti: one per exercise date.

    The output is a sigmoid level. The final bias starts at +2.0 (b ~ 0.88),
    so the untrained policy spends fuel slowly.
    """

    def __init__(self, dim, hidden, K_strike, payoff_fn):
        super().__init__()
        self.K = K_strike
        self.payoff_fn = payoff_fn
        layers, width_in = [], dim + 1
        for h in hidden:
            layers += [nn.Linear(width_in, h), nn.ReLU()]
            width_in = h
        head = nn.Linear(width_in, 1)
        nn.init.constant_(head.bias, 2.0)
        layers += [head]
        self.net = nn.Sequential(*layers)

    def forward(self, S):
        g = self.payoff_fn(S, self.K)
        x = torch.cat([S / self.K, (g / self.K).unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


class StopNet(nn.Module):
    """Stopping logit for DOS: one per exercise date.

    forward() returns the logit. Training uses the soft stopping probability
    sigmoid(logit); execution uses the hard decision logit >= 0 (Becker et
    al. 2019, eq. 8). Unlike the other networks, this one uses batch
    normalisation (on the input and before each activation), as DOS does;
    without it the method falls well short of its published values.
    Execution uses the running statistics.
    """

    def __init__(self, dim, hidden, K_strike, payoff_fn):
        super().__init__()
        self.K = K_strike
        self.payoff_fn = payoff_fn
        layers, width_in = [nn.BatchNorm1d(dim + 1)], dim + 1
        for h in hidden:
            layers += [nn.Linear(width_in, h), nn.BatchNorm1d(h), nn.ReLU()]
            width_in = h
        layers += [nn.Linear(width_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, S):                        # S: (batch, d) -> logit (batch,)
        g = self.payoff_fn(S, self.K)
        x = torch.cat([S / self.K, (g / self.K).unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


class GradNet(nn.Module):
    """Gradient head Z ~ sigma^T D_x u for RDBDP: one per exercise date.

    The output has d components, scaled by the strike.
    """

    def __init__(self, dim, hidden, K_strike, payoff_fn):
        super().__init__()
        self.K = K_strike
        self.payoff_fn = payoff_fn
        layers, width_in = [], dim + 1
        for h in hidden:
            layers += [nn.Linear(width_in, h), nn.ReLU()]
            width_in = h
        layers += [nn.Linear(width_in, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, S):                        # S: (batch, d) -> Z (batch, d)
        g = self.payoff_fn(S, self.K)
        x = torch.cat([S / self.K, (g / self.K).unsqueeze(-1)], dim=-1)
        return self.K * self.net(x)
