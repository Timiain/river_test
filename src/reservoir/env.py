from __future__ import annotations

import numpy as np

from .config import ReservoirConfig


class ReservoirSystem:
    def __init__(self, config: ReservoirConfig):
        self.cfg = config
        self.dt = config.dt_hours * 3600.0

    def head(self, storage: float) -> float:
        ratio = (storage - self.cfg.storage_min) / (self.cfg.storage_max - self.cfg.storage_min + 1e-9)
        return self.cfg.head_ref * (0.7 + 0.6 * np.clip(ratio, 0, 1))

    def step(self, storage: float, qin: float, q_turb: np.ndarray, q_gate: np.ndarray) -> dict:
        q_turb = np.clip(q_turb, self.cfg.turbine_q_min, self.cfg.turbine_q_max)
        q_gate = np.clip(q_gate, 0.0, self.cfg.gate_q_max)
        release = float(np.sum(q_turb) + np.sum(q_gate))
        release = float(np.clip(release, self.cfg.release_min, self.cfg.release_max))
        storage_next = storage + (qin - release) * self.dt
        storage_next = float(np.clip(storage_next, self.cfg.storage_min, self.cfg.storage_max))
        h = self.head(storage)
        power = float(self.cfg.power_coef * np.sum(q_turb) * h * self.cfg.dt_hours)
        return {
            "storage_next": storage_next,
            "release": release,
            "power": power,
            "head": h,
        }

    def objective(self, release: np.ndarray, power: np.ndarray, storage_end: float, gate_moves: float) -> float:
        flood = float(np.mean(release**2))
        p = float(np.sum(power))
        eco_pen = float(np.mean(np.maximum(release - 10000.0, 0)))
        level_pen = float((storage_end - self.cfg.storage_target_end) ** 2 / (1e8**2))
        return (
            self.cfg.flood_weight * flood
            - self.cfg.power_weight * p
            + self.cfg.eco_weight * eco_pen
            + self.cfg.level_weight * level_pen
            + self.cfg.gate_move_penalty * gate_moves
        )
