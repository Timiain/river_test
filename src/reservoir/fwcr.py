from __future__ import annotations

import numpy as np


def non_flood_rule(storage: float, storage_target: float, qin: float, n_units: int, n_gates: int, turbine_q_max: float):
    retreat = np.clip((storage - storage_target) / (storage_target + 1e-9), -0.2, 0.2)
    release = max(800.0, qin * 0.9 + retreat * 3500)
    q_turb_total = min(release, n_units * turbine_q_max)
    q_gate_total = max(0.0, release - q_turb_total)
    q_turb = np.full(n_units, q_turb_total / n_units)
    q_gate = np.full(n_gates, q_gate_total / n_gates)
    return q_turb, q_gate
