#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoir.baselines import dp_baseline, rule_baseline
from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.eval import summarize_run
from reservoir.gpr import RunoffGPRForecaster
from reservoir.mpc import MPCScheduler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="experiments/runoff.csv")
    p.add_argument("--outdir", type=str, default="experiments/results")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    qin = df["qin"].values.astype(float)
    split = int(len(qin) * 0.7)
    train, test = qin[:split], qin[split:]

    forecaster = RunoffGPRForecaster(kernel_type="se")
    report = forecaster.evaluate(train)
    forecaster.fit(train)

    cfg = ReservoirConfig()
    mcfg = MPCConfig(pred_horizon_h=36, step_h=2, feedback_k=1.0)
    mpc = MPCScheduler(cfg, mcfg, forecaster)

    out_rule = rule_baseline(cfg, test)
    out_dp = dp_baseline(cfg, test)
    out_mpc = mpc.run(test, train[-max(forecaster.selected_lags) - 2 :])

    rows = [
        summarize_run("Rule", test, out_rule),
        summarize_run("DP", test, out_dp),
        summarize_run("HOSM+FWCR+MPC", test, out_mpc),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "metrics.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(test[: len(out_mpc["release"])], label="Inflow")
    plt.plot(out_mpc["release"], label="MPC Release")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mpc_release.png", dpi=160)

    with open(outdir / "gpr_report.txt", "w", encoding="utf-8") as f:
        f.write(str(report))
    print(metrics)
    print("Saved to", outdir)


if __name__ == "__main__":
    main()
