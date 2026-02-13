from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.data import synthetic_runoff_series
from reservoir.gpr import RunoffGPRForecaster
from reservoir.mpc import MPCScheduler


def test_pipeline_smoke():
    q = synthetic_runoff_series(300, seed=0)["qin"].values
    split = 200
    gpr = RunoffGPRForecaster()
    gpr.fit(q[:split])
    out = MPCScheduler(ReservoirConfig(), MPCConfig(pred_horizon_h=12, step_h=2), gpr).run(q[split:], q[split-60:split])
    assert len(out["release"]) > 0
