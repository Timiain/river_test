from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HydrologySimulatorConfig:
    catchment_area_km2: float = 8600.0
    dt_hours: float = 1.0
    cn_runoff_coeff: float = 0.38
    soil_max_mm: float = 180.0
    percolation_coeff: float = 0.03
    baseflow_recession: float = 0.96
    channel_reservoir_coeff: float = 0.18


class ConceptualHydrologySimulator:
    """A compact rainfall-runoff simulator with soil store + baseflow + channel routing."""

    def __init__(self, cfg: HydrologySimulatorConfig | None = None, seed: int = 0):
        self.cfg = cfg or HydrologySimulatorConfig()
        self.rng = np.random.default_rng(seed)

    def _synthetic_meteo(self, n_steps: int, scenario: str) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_steps)
        precip = np.maximum(0.0, self.rng.gamma(shape=1.6, scale=2.2, size=n_steps) - 1.1)

        # storm pulses
        for c in self.rng.choice(n_steps, size=max(4, n_steps // 260), replace=False):
            width = int(self.rng.integers(4, 16))
            amp = float(self.rng.uniform(8, 34))
            idx = np.arange(max(0, c - 2 * width), min(n_steps, c + 2 * width))
            precip[idx] += amp * np.exp(-((idx - c) ** 2) / (2 * width**2))

        temp = 13 + 9 * np.sin(2 * np.pi * t / 720) + self.rng.normal(0, 1.5, size=n_steps)
        et0 = np.clip(0.06 * np.maximum(temp + 3.0, 0.0), 0.05, 1.3)

        if scenario == "wet":
            precip *= 1.25
        elif scenario == "dry":
            precip *= 0.72
            et0 *= 1.15
        elif scenario == "extreme":
            precip *= 1.15
            precip += 6.0 * np.maximum(0, np.sin(2 * np.pi * t / 96))
        elif scenario == "climate_trend":
            precip *= 1.0 + 0.20 * (t / max(1, n_steps - 1))
            et0 *= 1.0 + 0.05 * (t / max(1, n_steps - 1))
        return precip, et0

    def simulate(self, n_steps: int, scenario: str = "mixed") -> pd.DataFrame:
        precip, et0 = self._synthetic_meteo(n_steps, scenario)
        cfg = self.cfg

        soil = cfg.soil_max_mm * 0.65
        gw = 22.0
        ch_store = 0.0

        qin = np.zeros(n_steps)
        soil_arr = np.zeros(n_steps)
        quick_arr = np.zeros(n_steps)
        base_arr = np.zeros(n_steps)

        area_factor = cfg.catchment_area_km2 * 1000.0 / (cfg.dt_hours * 3600.0)  # mm/h -> m3/s

        for t in range(n_steps):
            p = precip[t]
            et = min(et0[t], soil * 0.03)
            soil = max(0.0, soil - et)

            runoff_excess = max(0.0, p - (1 - cfg.cn_runoff_coeff) * 5.0)
            infil = max(0.0, p - runoff_excess)
            soil = min(cfg.soil_max_mm, soil + infil)

            percolation = cfg.percolation_coeff * max(0.0, soil - 0.55 * cfg.soil_max_mm)
            soil = max(0.0, soil - percolation)

            gw = cfg.baseflow_recession * gw + percolation
            quick = runoff_excess
            base = 0.22 * gw

            inflow_mm = quick + base
            ch_store = (1 - cfg.channel_reservoir_coeff) * ch_store + cfg.channel_reservoir_coeff * inflow_mm

            qin[t] = max(50.0, ch_store * area_factor)
            soil_arr[t] = soil
            quick_arr[t] = quick * area_factor
            base_arr[t] = base * area_factor

        return pd.DataFrame(
            {
                "t": np.arange(n_steps),
                "precip_mm_h": precip,
                "et0_mm_h": et0,
                "soil_mm": soil_arr,
                "quickflow": quick_arr,
                "baseflow": base_arr,
                "qin": qin,
            }
        )


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


def hydrology_simulation_series(n_steps: int, seed: int = 0, scenario: str = "mixed") -> pd.DataFrame:
    sim = ConceptualHydrologySimulator(seed=seed)
    return sim.simulate(n_steps=n_steps, scenario=scenario)


def make_lag_features(series: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(lags)
    x, y = [], []
    for i in range(max_lag, len(series)):
        x.append([series[i - l] for l in lags])
        y.append(series[i])
    return np.asarray(x), np.asarray(y)
