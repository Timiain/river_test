from __future__ import annotations

import numpy as np

from .config import ReservoirConfig
from .env import ReservoirSystem


def rule_baseline(cfg: ReservoirConfig, qin: np.ndarray) -> dict:
    sys = ReservoirSystem(cfg)
    s = cfg.storage_init
    rel, pwr, st = [], [], [s]
    for q in qin:
        target_rel = np.clip(0.9 * q + 500, cfg.release_min, cfg.release_max)
        qt = min(target_rel, cfg.n_units * cfg.turbine_q_max)
        qg = max(0.0, target_rel - qt)
        step = sys.step(s, q, np.full(cfg.n_units, qt / cfg.n_units), np.full(cfg.n_gates, qg / cfg.n_gates))
        s = step["storage_next"]
        rel.append(step["release"])
        pwr.append(step["power"])
        st.append(s)
    return {"release": np.array(rel), "power": np.array(pwr), "storage": np.array(st)}


def dp_baseline(cfg: ReservoirConfig, qin: np.ndarray, n_grid: int = 25) -> dict:
    sys = ReservoirSystem(cfg)
    T = len(qin)
    grid = np.linspace(cfg.storage_min, cfg.storage_max, n_grid)
    val = np.full((T + 1, n_grid), np.inf)
    act = np.zeros((T, n_grid))
    val[T, :] = ((grid - cfg.storage_target_end) / 1e8) ** 2
    actions = np.linspace(cfg.release_min, cfg.release_max, 12)
    for t in range(T - 1, -1, -1):
        for i, s in enumerate(grid):
            best_v, best_a = np.inf, actions[0]
            for a in actions:
                s2 = np.clip(s + (qin[t] - a) * cfg.dt_hours * 3600, cfg.storage_min, cfg.storage_max)
                j = int(np.argmin(np.abs(grid - s2)))
                flood = a**2
                power = cfg.power_coef * min(a, cfg.n_units * cfg.turbine_q_max) * sys.head(s) * cfg.dt_hours
                cost = cfg.flood_weight * flood - cfg.power_weight * power + val[t + 1, j]
                if cost < best_v:
                    best_v, best_a = cost, a
            val[t, i], act[t, i] = best_v, best_a

    s = cfg.storage_init
    rel, pwr, st = [], [], [s]
    for t in range(T):
        i = int(np.argmin(np.abs(grid - s)))
        a = act[t, i]
        qt = min(a, cfg.n_units * cfg.turbine_q_max)
        qg = max(0.0, a - qt)
        step = sys.step(s, qin[t], np.full(cfg.n_units, qt / cfg.n_units), np.full(cfg.n_gates, qg / cfg.n_gates))
        s = step["storage_next"]
        rel.append(step["release"])
        pwr.append(step["power"])
        st.append(s)
    return {"release": np.array(rel), "power": np.array(pwr), "storage": np.array(st)}
