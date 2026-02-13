#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reservoir.product import preflight_environment, product_presets, sales_readiness_summary


def main():
    print("=== Product Readiness Audit ===")
    print("\n[1] Environment checks")
    env = preflight_environment()
    for k, v in env.items():
        print(f"- {k:12s}: {'OK' if v else 'MISSING'}")

    print("\n[2] Sales readiness summary")
    for k, v in sales_readiness_summary().items():
        print(f"- {k}: {v}")

    print("\n[3] Recommended presets")
    for p in product_presets():
        print(f"- {p.name}: scenario={p.scenario}, horizon={p.pred_h}, step={p.step_h} ({p.description})")


if __name__ == "__main__":
    main()
