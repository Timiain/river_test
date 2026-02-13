#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from reservoir.data import hydrology_simulation_series


def main():
    p = argparse.ArgumentParser(description="Run conceptual hydrology simulator and export inflow dataset")
    p.add_argument("--n_steps", type=int, default=24 * 120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", type=str, default="mixed", choices=["mixed", "wet", "dry", "extreme", "climate_trend"])
    p.add_argument("--out", type=str, default="experiments/hydro_sim.csv")
    args = p.parse_args()

    df = hydrology_simulation_series(args.n_steps, seed=args.seed, scenario=args.scenario)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(df["precip_mm_h"], label="Precip(mm/h)")
    ax[0].plot(df["et0_mm_h"], label="ET0(mm/h)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(df["qin"], label="Inflow(m3/s)", color="black")
    ax[1].plot(df["quickflow"], label="Quickflow", alpha=0.8)
    ax[1].plot(df["baseflow"], label="Baseflow", alpha=0.8)
    ax[1].legend(ncol=3)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'), dpi=160)

    print(f"saved simulator data: {out}")
    print(f"saved simulator figure: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
