from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import dp_baseline, rule_baseline
from .config import MPCConfig, ReservoirConfig
from .data import hydrology_simulation_series, synthetic_runoff_series
from .eval import summarize_run
from .gpr import RunoffGPRForecaster
from .mpc import MPCScheduler

ALLOWED_SCENARIOS = {"mixed", "wet", "dry", "extreme", "climate_trend", "hydro_physics"}


@dataclass
class ExperimentBundle:
    data: pd.DataFrame
    metrics: pd.DataFrame
    series: dict[str, dict]
    gpr_report: object


def generate_synthetic_dataset(n_steps: int, seed: int, scenario: str = "mixed") -> pd.DataFrame:
    if n_steps < 120:
        raise ValueError("n_steps should be >= 120 for stable train/test and lag features")
    if scenario not in ALLOWED_SCENARIOS:
        raise ValueError(f"unsupported scenario={scenario}; expected one of {sorted(ALLOWED_SCENARIOS)}")

    if scenario == "hydro_physics":
        return hydrology_simulation_series(n_steps=n_steps, seed=seed, scenario="mixed")

    df = synthetic_runoff_series(n_steps=n_steps, seed=seed)
    q = df["qin"].to_numpy()
    t = np.arange(len(q))
    if scenario == "wet":
        q = q * 1.2 + 1200
    elif scenario == "dry":
        q = np.maximum(200, q * 0.78)
    elif scenario == "extreme":
        q = q + 4200 * np.maximum(0, np.sin(2 * np.pi * t / 72))
    elif scenario == "climate_trend":
        q = q * (1.0 + 0.18 * (t / max(1, len(t) - 1)))
    df["qin"] = q
    return df


def run_benchmark(df: pd.DataFrame, cfg: ReservoirConfig | None = None, mpc_cfg: MPCConfig | None = None) -> ExperimentBundle:
    if "qin" not in df.columns:
        raise ValueError("input dataframe must include 'qin' column")

    cfg = cfg or ReservoirConfig()
    mpc_cfg = mpc_cfg or MPCConfig(pred_horizon_h=36, step_h=2, feedback_k=1.0)

    qin = df["qin"].to_numpy(dtype=float)
    if len(qin) < 180:
        raise ValueError("dataset too short; need at least 180 steps")

    split = int(len(qin) * 0.7)
    train, test = qin[:split], qin[split:]

    forecaster = RunoffGPRForecaster(kernel_type="se")
    gpr_report = forecaster.evaluate(train)
    forecaster.fit(train)

    timings = {}

    t0 = time.perf_counter()
    out_rule = rule_baseline(cfg, test)
    timings["Rule"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_dp = dp_baseline(cfg, test)
    timings["DP"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_mpc = MPCScheduler(cfg, mpc_cfg, forecaster).run(test, train[-max(forecaster.selected_lags) - 2 :])
    timings["HOSM+FWCR+MPC"] = time.perf_counter() - t0

    rows = [
        summarize_run("Rule", test, out_rule),
        summarize_run("DP", test, out_dp),
        summarize_run("HOSM+FWCR+MPC", test, out_mpc),
    ]
    metrics = pd.DataFrame(rows)
    metrics["runtime_s"] = metrics["method"].map(timings)

    series = {
        "qin_test": test,
        "Rule": out_rule,
        "DP": out_dp,
        "HOSM+FWCR+MPC": out_mpc,
    }
    return ExperimentBundle(data=df, metrics=metrics, series=series, gpr_report=gpr_report)
