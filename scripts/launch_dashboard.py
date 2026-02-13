#!/usr/bin/env python
"""Python entrypoint to launch the Streamlit dashboard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def main():
    if importlib.util.find_spec("streamlit") is None:
        print("[ERROR] streamlit is not installed. Please run: pip install -r requirements.txt")
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
