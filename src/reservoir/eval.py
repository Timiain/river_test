from __future__ import annotations

import numpy as np


def peak_shaving_rate(qin: np.ndarray, release: np.ndarray) -> float:
    return float((np.max(qin) - np.max(release)) / np.max(qin))


def summarize_run(name: str, qin: np.ndarray, out: dict) -> dict:
    rel = out["release"]
    pwr = out["power"]
    st = out["storage"]
    return {
        "method": name,
        "peak_shaving": peak_shaving_rate(qin[: len(rel)], rel),
        "energy": float(np.sum(pwr)),
        "end_storage": float(st[-1]),
        "max_release": float(np.max(rel)),
    }
