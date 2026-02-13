#!/usr/bin/env python
import argparse
import itertools
from pathlib import Path

import pandas as pd

from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.eval import summarize_run
from reservoir.gpr import RunoffGPRForecaster
from reservoir.mpc import MPCScheduler


def main():
    p = argparse.ArgumentParser(description="Sensitivity analysis for MPC horizon/step/FWCR threshold")
    p.add_argument("--data", type=str, default="experiments/runoff.csv")
    p.add_argument("--out", type=str, default="experiments/results/sensitivity.csv")
    args = p.parse_args()

    data = pd.read_csv(args.data)
    q = data["qin"].values
    split = int(0.7 * len(q))
    train, test = q[:split], q[split:]

    gpr = RunoffGPRForecaster(kernel_type="se")
    gpr.fit(train)

    rows = []
    for ph, step, th in itertools.product([24, 36, 48], [1, 2, 3], [12000, 14000, 16000]):
        cfg = ReservoirConfig(fwcr_threshold=th)
        mcfg = MPCConfig(pred_horizon_h=ph, step_h=step, feedback_k=1.0)
        out = MPCScheduler(cfg, mcfg, gpr).run(test, train[-max(gpr.selected_lags) - 2 :])
        s = summarize_run("mpc", test, out)
        s.update({"pred_h": ph, "step_h": step, "threshold": th})
        rows.append(s)

    res = pd.DataFrame(rows).sort_values("peak_shaving", ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(res.head(10))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
