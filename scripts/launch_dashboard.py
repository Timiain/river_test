#!/usr/bin/env python
"""Python entrypoint to launch the Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
