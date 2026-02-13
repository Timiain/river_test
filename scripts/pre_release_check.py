#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], with_src_path: bool = False) -> tuple[int, str]:
    env = None
    if with_src_path:
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = "src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-release checks for reservoir product")
    p.add_argument("--strict", action="store_true", help="return non-zero if any check fails")
    args = p.parse_args()

    checks = [
        ("compile", [sys.executable, "-m", "compileall", "src", "scripts", "app.py", "tests"]),
        ("readiness", [sys.executable, "scripts/product_readiness_audit.py"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
    ]

    failed = 0
    print("=== Pre-release check ===")
    for name, cmd in checks:
        code, out = run(cmd, with_src_path=(name == "pytest"))
        ok = code == 0
        if not ok:
            failed += 1
        print(f"\n[{name}] {'PASS' if ok else 'WARN'}: {' '.join(cmd)}")
        if out:
            print(out[:1200])

    report = Path("experiments/results/pre_release_report.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"failed_checks={failed}\nstrict={args.strict}\n")
    print(f"\nSaved report: {report}")

    if args.strict and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
