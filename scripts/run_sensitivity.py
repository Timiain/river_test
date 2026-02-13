#!/usr/bin/env python
import itertools
from pathlib import Path

import pandas as pd

from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.eval import summarize_run
from reservoir.gpr import RunoffGPRForecaster
from reservoir.mpc import MPCScheduler


def main():
    data = pd.read_csv("experiments/runoff.csv")
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
    res = pd.DataFrame(rows)
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    res.to_csv("experiments/results/sensitivity.csv", index=False)
    print(res.sort_values("peak_shaving", ascending=False).head(10))


if __name__ == "__main__":
    main()
