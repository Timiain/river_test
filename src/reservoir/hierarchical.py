from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from .config import ReservoirConfig, MPCConfig
from .env import ReservoirSystem


class HierarchicalOptimizer:
    def __init__(self, cfg: ReservoirConfig, mpc_cfg: MPCConfig):
        self.cfg = cfg
        self.mpc_cfg = mpc_cfg
        self.sys = ReservoirSystem(cfg)

    def _simulate_upper(self, x: np.ndarray, storage0: float, qin: np.ndarray):
        h = len(qin)
        x = x.reshape(h, 2)
        qt_total = np.clip(x[:, 0], self.cfg.n_units * self.cfg.turbine_q_min, self.cfg.n_units * self.cfg.turbine_q_max)
        qg_total = np.clip(x[:, 1], 0, self.cfg.n_gates * self.cfg.gate_q_max)
        storage = storage0
        rel, power = [], []
        for t in range(h):
            q_turb = np.full(self.cfg.n_units, qt_total[t] / self.cfg.n_units)
            q_gate = np.full(self.cfg.n_gates, qg_total[t] / self.cfg.n_gates)
            step = self.sys.step(storage, qin[t], q_turb, q_gate)
            storage = step["storage_next"]
            rel.append(step["release"])
            power.append(step["power"])
        return np.asarray(rel), np.asarray(power), storage, qt_total, qg_total

    def _lower_allocate(self, qg_total: np.ndarray, prev_gate: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        prev_gate = np.zeros(self.cfg.n_gates) if prev_gate is None else prev_gate.copy()
        gate = np.zeros((len(qg_total), self.cfg.n_gates))
        moves = 0.0
        for t, q in enumerate(qg_total):
            rem = q
            rank = np.argsort(prev_gate)
            cur = np.zeros(self.cfg.n_gates)
            for idx in rank:
                alloc = min(self.cfg.gate_q_max, rem)
                cur[idx] = alloc
                rem -= alloc
                if rem <= 1e-9:
                    break
            moves += np.sum(np.abs((cur > 1e-6).astype(float) - (prev_gate > 1e-6).astype(float)))
            gate[t] = cur
            prev_gate = cur
        return gate, float(moves)

    def optimize(self, storage0: float, qin: np.ndarray, prev_gate: np.ndarray | None = None) -> dict:
        h = len(qin)
        bounds = []
        for _ in range(h):
            bounds.append((self.cfg.n_units * self.cfg.turbine_q_min, self.cfg.n_units * self.cfg.turbine_q_max))
            bounds.append((0.0, self.cfg.n_gates * self.cfg.gate_q_max))

        def fitness(x):
            release, power, s_end, _, qg = self._simulate_upper(x, storage0, qin)
            _, moves = self._lower_allocate(qg, prev_gate)
            return self.sys.objective(release, power, s_end, moves)

        result = differential_evolution(
            fitness,
            bounds=bounds,
            maxiter=self.mpc_cfg.de_maxiter,
            popsize=self.mpc_cfg.de_popsize,
            polish=False,
            disp=False,
            seed=1,
        )
        release, power, s_end, qt, qg = self._simulate_upper(result.x, storage0, qin)
        gates, moves = self._lower_allocate(qg, prev_gate)
        q_turb = np.repeat((qt / self.cfg.n_units)[:, None], self.cfg.n_units, axis=1)
        return {
            "q_turb": q_turb,
            "q_gate": gates,
            "release": release,
            "power": power,
            "storage_end": s_end,
            "gate_moves": moves,
            "cost": fitness(result.x),
        }
