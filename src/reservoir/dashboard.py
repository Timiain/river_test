from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import MPCConfig, ReservoirConfig
from .experiment import ExperimentBundle, generate_synthetic_dataset, run_benchmark
from .rl import ReservoirRLTrainer


@dataclass
class PanelConfig:
    n_steps: int = 24 * 90
    seed: int = 42
    scenario: str = "mixed"
    fwcr_threshold: float = 14000.0
    pred_h: int = 36
    step_h: int = 2


def run_panel_experiment(panel_cfg: PanelConfig) -> ExperimentBundle:
    cfg = ReservoirConfig(fwcr_threshold=panel_cfg.fwcr_threshold)
    mcfg = MPCConfig(pred_horizon_h=panel_cfg.pred_h, step_h=panel_cfg.step_h, feedback_k=1.0)
    df = generate_synthetic_dataset(panel_cfg.n_steps, panel_cfg.seed, panel_cfg.scenario)
    return run_benchmark(df, cfg=cfg, mpc_cfg=mcfg)


def run_rl_training(seed: int, epochs: int, horizon: int, outdir: str = "experiments/results") -> str:
    trainer = ReservoirRLTrainer(ReservoirConfig(), seed=seed)
    policy = trainer.train_ppo(epochs=epochs, horizon=horizon)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"ppo_policy_seed{seed}_ep{epochs}.pt"
    import torch

    torch.save(policy.state_dict(), model_path)
    return str(model_path)


def render_release_figure(bundle: ExperimentBundle):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bundle.series["qin_test"], label="Inflow", color="black", linewidth=1.8)
    ax.plot(bundle.series["Rule"]["release"], label="Rule", alpha=0.9)
    ax.plot(bundle.series["DP"]["release"], label="DP", alpha=0.9)
    ax.plot(bundle.series["HOSM+FWCR+MPC"]["release"], label="HOSM+FWCR+MPC", linewidth=2.2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Flow (m³/s)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4)
    return fig


def render_storage_figure(bundle: ExperimentBundle):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bundle.series["Rule"]["storage"], label="Rule", alpha=0.9)
    ax.plot(bundle.series["DP"]["storage"], label="DP", alpha=0.9)
    ax.plot(bundle.series["HOSM+FWCR+MPC"]["storage"], label="HOSM+FWCR+MPC", linewidth=2.2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Storage (m³)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
    return fig


def build_growth_table(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics.set_index("method")
    rule_peak = base.loc["Rule", "peak_shaving"]
    rule_energy = base.loc["Rule", "energy"]
    x = metrics.copy()
    x["peak_growth_vs_rule_%"] = (x["peak_shaving"] - rule_peak) / (abs(rule_peak) + 1e-9) * 100
    x["energy_growth_vs_rule_%"] = (x["energy"] - rule_energy) / (abs(rule_energy) + 1e-9) * 100
    return x
