#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.experiment import run_benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="experiments/runoff.csv")
    p.add_argument("--outdir", type=str, default="experiments/results")
    p.add_argument("--fwcr_threshold", type=float, default=14000.0)
    p.add_argument("--pred_h", type=int, default=36)
    p.add_argument("--step_h", type=int, default=2)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    cfg = ReservoirConfig(fwcr_threshold=args.fwcr_threshold)
    mcfg = MPCConfig(pred_horizon_h=args.pred_h, step_h=args.step_h, feedback_k=1.0)
    bundle = run_benchmark(df, cfg=cfg, mpc_cfg=mcfg)

    bundle.metrics.to_csv(outdir / "metrics.csv", index=False)

    qin_test = bundle.series["qin_test"]
    plt.figure(figsize=(10, 4))
    plt.plot(qin_test, label="Inflow")
    plt.plot(bundle.series["HOSM+FWCR+MPC"]["release"], label="MPC Release")
    plt.plot(bundle.series["DP"]["release"], label="DP Release", alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "release_compare.png", dpi=160)

    with open(outdir / "gpr_report.txt", "w", encoding="utf-8") as f:
        f.write(str(bundle.gpr_report))

    print(bundle.metrics)
    print("Saved to", outdir)


if __name__ == "__main__":
    main()
