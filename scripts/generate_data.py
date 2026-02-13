#!/usr/bin/env python
import argparse
from pathlib import Path

from reservoir.experiment import generate_synthetic_dataset


def main():
    p = argparse.ArgumentParser(description="Generate synthetic runoff data for reservoir scheduling experiments")
    p.add_argument("--n_steps", type=int, default=24 * 120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", type=str, default="mixed", choices=["mixed", "wet", "dry", "extreme", "climate_trend"])
    p.add_argument("--out", type=str, default="experiments/runoff.csv")
    args = p.parse_args()

    df = generate_synthetic_dataset(args.n_steps, args.seed, scenario=args.scenario)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out} with {len(df)} rows; scenario={args.scenario}")


if __name__ == "__main__":
    main()
