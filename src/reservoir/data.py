from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_runoff_series(n_steps: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    seasonal = 5500 + 2500 * np.sin(2 * np.pi * t / 168)
    storms = np.zeros(n_steps)
    for c in rng.choice(n_steps, size=max(3, n_steps // 220), replace=False):
        width = rng.integers(6, 20)
        amp = rng.uniform(4000, 13000)
        idx = np.arange(max(0, c - 2 * width), min(n_steps, c + 2 * width))
        storms[idx] += amp * np.exp(-((idx - c) ** 2) / (2 * width**2))
    noise = rng.normal(0, 450, size=n_steps)
    q = np.clip(seasonal + storms + noise, 300, None)
    return pd.DataFrame({"t": t, "qin": q})


def make_lag_features(series: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(lags)
    x, y = [], []
    for i in range(max_lag, len(series)):
        x.append([series[i - l] for l in lags])
        y.append(series[i])
    return np.asarray(x), np.asarray(y)
