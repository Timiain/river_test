from dataclasses import dataclass


@dataclass
class ReservoirConfig:
    dt_hours: float = 1.0
    horizon: int = 48
    n_units: int = 3
    n_gates: int = 4
    storage_min: float = 1.0e8
    storage_max: float = 3.0e8
    storage_init: float = 2.0e8
    storage_target_end: float = 1.9e8
    release_min: float = 800.0
    release_max: float = 22000.0
    turbine_q_min: float = 300.0
    turbine_q_max: float = 4000.0
    gate_q_max: float = 6000.0
    gate_move_penalty: float = 1.0
    flood_weight: float = 1.0
    power_weight: float = 0.4
    eco_weight: float = 0.2
    level_weight: float = 6.0
    power_coef: float = 0.0025
    head_ref: float = 55.0
    fwcr_threshold: float = 14000.0


@dataclass
class MPCConfig:
    pred_horizon_h: int = 36
    step_h: int = 2
    feedback_k: float = 1.0
    de_maxiter: int = 40
    de_popsize: int = 10
