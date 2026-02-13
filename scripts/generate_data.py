#!/usr/bin/env python
import argparse
from pathlib import Path

from reservoir.data import synthetic_runoff_series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps", type=int, default=24 * 120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="experiments/runoff.csv")
    args = p.parse_args()

    df = synthetic_runoff_series(args.n_steps, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out} with {len(df)} rows")


if __name__ == "__main__":
    main()
