from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ReservoirConfig
from .env import ReservoirSystem


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.log_std.exp(), self.v(h)


class ReservoirRLTrainer:
    def __init__(self, cfg: ReservoirConfig, seed: int = 0):
        self.cfg = cfg
        self.sys = ReservoirSystem(cfg)
        self.rng = np.random.default_rng(seed)

    def sample_episode(self, policy: PolicyNet, horizon: int = 72):
        s = self.cfg.storage_init
        obs, acts, logps, vals, rews = [], [], [], [], []
        for t in range(horizon):
            qin = 6000 + 3500 * np.sin(2 * np.pi * t / 24) + self.rng.normal(0, 900)
            o = np.array([s / self.cfg.storage_max, qin / 20000, t / horizon], dtype=np.float32)
            ot = torch.tensor(o).unsqueeze(0)
            mu, std, v = policy(ot)
            dist = torch.distributions.Normal(mu, std)
            at = dist.sample()
            a = at.detach().numpy()[0]
            qt = np.clip((a[0] + 1) / 2, 0, 1) * self.cfg.n_units * self.cfg.turbine_q_max
            qg = np.clip((a[1] + 1) / 2, 0, 1) * self.cfg.n_gates * self.cfg.gate_q_max
            step = self.sys.step(
                s,
                qin,
                np.full(self.cfg.n_units, qt / self.cfg.n_units),
                np.full(self.cfg.n_gates, qg / self.cfg.n_gates),
            )
            s = step["storage_next"]
            reward = -(
                1e-8 * step["release"] ** 2
                - 0.3 * step["power"]
                + 5.0 * ((s - self.cfg.storage_target_end) / 1e8) ** 2
            )
            obs.append(o)
            acts.append(a)
            logps.append(dist.log_prob(at).sum())
            vals.append(v.squeeze())
            rews.append(float(reward))
        return obs, acts, logps, vals, rews

    def train_ppo(self, epochs: int = 80, horizon: int = 72, gamma: float = 0.98):
        policy = PolicyNet(3, 2)
        opt = optim.Adam(policy.parameters(), lr=3e-4)
        for _ in range(epochs):
            obs, acts, logps, vals, rews = self.sample_episode(policy, horizon)
            returns, g = [], 0.0
            for r in rews[::-1]:
                g = r + gamma * g
                returns.append(g)
            returns = torch.tensor(returns[::-1], dtype=torch.float32)
            vals = torch.stack(vals)
            adv = (returns - vals.detach())
            loss_p = -(torch.stack(logps) * adv).mean()
            loss_v = ((vals - returns) ** 2).mean()
            loss = loss_p + 0.5 * loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
        return policy
