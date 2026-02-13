from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class ProductPreset:
    name: str
    n_steps: int
    scenario: str
    fwcr_threshold: float
    pred_h: int
    step_h: int
    description: str


def product_presets() -> list[ProductPreset]:
    return [
        ProductPreset(
            name="演示默认（平衡）",
            n_steps=24 * 90,
            scenario="mixed",
            fwcr_threshold=14000.0,
            pred_h=36,
            step_h=2,
            description="适合售前Demo，运行稳定、指标平衡。",
        ),
        ProductPreset(
            name="防洪优先（极端）",
            n_steps=24 * 120,
            scenario="extreme",
            fwcr_threshold=12000.0,
            pred_h=48,
            step_h=2,
            description="用于展示极端来水下削峰优势。",
        ),
        ProductPreset(
            name="物理过程（专家）",
            n_steps=24 * 90,
            scenario="hydro_physics",
            fwcr_threshold=14000.0,
            pred_h=36,
            step_h=2,
            description="采用概念水文模拟器，强调可解释性。",
        ),
    ]


def preflight_environment() -> dict[str, bool]:
    req = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "streamlit"]
    return {k: find_spec(k) is not None for k in req}


def sales_readiness_summary() -> dict[str, str]:
    env = preflight_environment()
    core_ready = all(env[k] for k in ["numpy", "pandas", "scipy", "sklearn", "matplotlib"])
    panel_ready = env["streamlit"]
    rl_ready = env["torch"]
    return {
        "核心算法依赖": "就绪" if core_ready else "缺失依赖",
        "可视化面板": "就绪" if panel_ready else "缺少streamlit",
        "强化学习训练": "就绪" if rl_ready else "缺少torch",
    }
