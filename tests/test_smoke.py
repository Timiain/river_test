import pytest

from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.experiment import generate_synthetic_dataset, run_benchmark
from reservoir.gpr import RunoffGPRForecaster
from reservoir.mpc import MPCScheduler


def test_pipeline_smoke():
    q = generate_synthetic_dataset(300, seed=0)["qin"].values
    split = 200
    gpr = RunoffGPRForecaster()
    gpr.fit(q[:split])
    out = MPCScheduler(ReservoirConfig(), MPCConfig(pred_horizon_h=12, step_h=2), gpr).run(q[split:], q[split - 60 : split])
    assert len(out["release"]) > 0
    assert "feedback_errors" in out


def test_benchmark_smoke():
    df = generate_synthetic_dataset(280, seed=1, scenario="extreme")
    bundle = run_benchmark(df, cfg=ReservoirConfig(), mpc_cfg=MPCConfig(pred_horizon_h=24, step_h=2))
    assert set(bundle.metrics["method"]) == {"Rule", "DP", "HOSM+FWCR+MPC"}


def test_data_validation():
    with pytest.raises(ValueError):
        generate_synthetic_dataset(60, seed=1)

    with pytest.raises(ValueError):
        generate_synthetic_dataset(240, seed=1, scenario="unknown")

    with pytest.raises(ValueError):
        run_benchmark(generate_synthetic_dataset(160, seed=1))
