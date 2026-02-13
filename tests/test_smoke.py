import importlib.util

import pytest

from reservoir.product import preflight_environment, product_presets


NUMPY_READY = importlib.util.find_spec("numpy") is not None


@pytest.mark.skipif(not NUMPY_READY, reason="numpy/scientific stack not installed in current environment")
def test_pipeline_smoke():
    from reservoir.config import MPCConfig, ReservoirConfig
    from reservoir.experiment import generate_synthetic_dataset
    from reservoir.gpr import RunoffGPRForecaster
    from reservoir.mpc import MPCScheduler

    q = generate_synthetic_dataset(300, seed=0)["qin"].values
    split = 200
    gpr = RunoffGPRForecaster()
    gpr.fit(q[:split])
    out = MPCScheduler(ReservoirConfig(), MPCConfig(pred_horizon_h=12, step_h=2), gpr).run(q[split:], q[split - 60 : split])
    assert len(out["release"]) > 0
    assert "feedback_errors" in out


@pytest.mark.skipif(not NUMPY_READY, reason="numpy/scientific stack not installed in current environment")
def test_benchmark_smoke():
    from reservoir.config import MPCConfig, ReservoirConfig
    from reservoir.experiment import generate_synthetic_dataset, run_benchmark

    df = generate_synthetic_dataset(280, seed=1, scenario="extreme")
    bundle = run_benchmark(df, cfg=ReservoirConfig(), mpc_cfg=MPCConfig(pred_horizon_h=24, step_h=2))
    assert set(bundle.metrics["method"]) == {"Rule", "DP", "HOSM+FWCR+MPC"}


@pytest.mark.skipif(not NUMPY_READY, reason="numpy/scientific stack not installed in current environment")
def test_hydrology_simulator_columns_and_range():
    from reservoir.data import hydrology_simulation_series

    df = hydrology_simulation_series(240, seed=7)
    expected = {"precip_mm_h", "et0_mm_h", "soil_mm", "quickflow", "baseflow", "qin"}
    assert expected.issubset(set(df.columns))
    assert float(df["qin"].min()) >= 0
    assert float(df["soil_mm"].min()) >= 0


@pytest.mark.skipif(not NUMPY_READY, reason="numpy/scientific stack not installed in current environment")
def test_data_validation():
    from reservoir.experiment import generate_synthetic_dataset, run_benchmark

    with pytest.raises(ValueError):
        generate_synthetic_dataset(60, seed=1)

    with pytest.raises(ValueError):
        generate_synthetic_dataset(240, seed=1, scenario="unknown")

    with pytest.raises(ValueError):
        run_benchmark(generate_synthetic_dataset(160, seed=1))


def test_product_presets_and_preflight_shape():
    presets = product_presets()
    assert len(presets) >= 3
    env = preflight_environment()
    assert "numpy" in env and "streamlit" in env
