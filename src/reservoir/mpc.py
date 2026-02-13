from __future__ import annotations

import numpy as np

from .config import MPCConfig, ReservoirConfig
from .env import ReservoirSystem
from .fwcr import non_flood_rule
from .hierarchical import HierarchicalOptimizer


class MPCScheduler:
    def __init__(self, cfg: ReservoirConfig, mpc_cfg: MPCConfig, forecaster):
        self.cfg = cfg
        self.mpc_cfg = mpc_cfg
        self.forecaster = forecaster
        self.sys = ReservoirSystem(cfg)
        self.opt = HierarchicalOptimizer(cfg, mpc_cfg)

    def run(self, qin_actual: np.ndarray, qin_init_hist: np.ndarray) -> dict:
        storage = self.cfg.storage_init
        gate_prev = np.zeros(self.cfg.n_gates)
        hist = qin_init_hist.astype(float).copy().tolist()
        releases, powers, storages, mode = [], [], [storage], []

        for k in range(0, len(qin_actual), self.mpc_cfg.step_h):
            horizon = min(self.mpc_cfg.pred_horizon_h, len(qin_actual) - k)
            qf = self.forecaster.rolling_predict(np.array(hist), horizon)
            if k > 0:
                err = qin_actual[k - 1] - hist[-1]
                qf = qf + self.mpc_cfg.feedback_k * err
                qf = np.clip(qf, 100.0, None)

            if qf[0] > self.cfg.fwcr_threshold:
                plan = self.opt.optimize(storage, qf, gate_prev)
                qt = plan["q_turb"][0]
                qg = plan["q_gate"][0]
                mode.append("flood_opt")
            else:
                qt, qg = non_flood_rule(
                    storage, self.cfg.storage_target_end, qf[0], self.cfg.n_units, self.cfg.n_gates, self.cfg.turbine_q_max
                )
                mode.append("non_flood_rule")

            for j in range(self.mpc_cfg.step_h):
                if k + j >= len(qin_actual):
                    break
                step = self.sys.step(storage, qin_actual[k + j], qt, qg)
                storage = step["storage_next"]
                releases.append(step["release"])
                powers.append(step["power"])
                storages.append(storage)
                hist.append(qin_actual[k + j])
                gate_prev = qg

        return {
            "release": np.array(releases),
            "power": np.array(powers),
            "storage": np.array(storages),
            "mode": mode,
        }
