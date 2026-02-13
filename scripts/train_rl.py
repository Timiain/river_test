#!/usr/bin/env python
from pathlib import Path

import torch

from reservoir.config import ReservoirConfig
from reservoir.rl import ReservoirRLTrainer


def main():
    trainer = ReservoirRLTrainer(ReservoirConfig())
    policy = trainer.train_ppo(epochs=50, horizon=72)
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), "experiments/results/ppo_policy.pt")
    print("saved PPO policy")


if __name__ == "__main__":
    main()
