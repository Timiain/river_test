from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from reservoir.config import MPCConfig, ReservoirConfig
from reservoir.experiment import generate_synthetic_dataset, run_benchmark


st.set_page_config(page_title="Reservoir HOSM-MPC Demo", layout="wide")
st.title("水库调度算法可视化展示系统（HOSM + FWCR + MPC）")
st.caption("一键生成模拟来水，自动训练预测模型并对比 Rule / DP / HOSM+FWCR+MPC 三类方法。")

with st.sidebar:
    st.header("模拟数据与实验配置")
    n_steps = st.slider("总时长（小时）", min_value=240, max_value=24 * 240, value=24 * 90, step=24)
    seed = st.number_input("随机种子", min_value=0, value=42, step=1)
    scenario = st.selectbox("水文情景", ["mixed", "wet", "dry", "extreme", "climate_trend"])
    fwcr_threshold = st.slider("FWCR阈值(m³/s)", min_value=8000, max_value=20000, value=14000, step=500)
    pred_h = st.select_slider("MPC预测时域(h)", options=[24, 36, 48], value=36)
    step_h = st.select_slider("MPC决策步长(h)", options=[1, 2, 3], value=2)

    run_btn = st.button("🚀 生成模拟数据并运行对比实验", type="primary")

if run_btn:
    cfg = ReservoirConfig(fwcr_threshold=float(fwcr_threshold))
    mcfg = MPCConfig(pred_horizon_h=int(pred_h), step_h=int(step_h), feedback_k=1.0)

    with st.spinner("正在生成模拟数据并运行算法..."):
        df = generate_synthetic_dataset(n_steps=int(n_steps), seed=int(seed), scenario=scenario)
        bundle = run_benchmark(df, cfg=cfg, mpc_cfg=mcfg)

    st.success("实验完成。")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("方法性能对比")
        show = bundle.metrics.copy()
        for col in ["peak_shaving", "energy", "runtime_s"]:
            show[col] = show[col].map(lambda x: float(f"{x:.4f}"))
        st.dataframe(show, use_container_width=True)

        best_peak = show.sort_values("peak_shaving", ascending=False).iloc[0]
        st.info(f"削峰率最佳方法：**{best_peak['method']}**，削峰率={best_peak['peak_shaving']:.4f}")

    with c2:
        st.subheader("GPR预测质量")
        rep = bundle.gpr_report
        st.write(
            {
                "selected_lags": rep.selected_lags,
                "RMSE": round(rep.rmse, 3),
                "R2": round(rep.r2, 3),
                "MAE": round(rep.mae, 3),
                "QR20": round(rep.qr20, 3),
            }
        )

    st.subheader("来水-下泄对比曲线")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bundle.series["qin_test"], label="Inflow", color="black", linewidth=1.5)
    ax.plot(bundle.series["Rule"]["release"], label="Rule")
    ax.plot(bundle.series["DP"]["release"], label="DP")
    ax.plot(bundle.series["HOSM+FWCR+MPC"]["release"], label="HOSM+FWCR+MPC", linewidth=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Flow (m³/s)")
    ax.legend(ncol=4)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("库容轨迹对比")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(bundle.series["Rule"]["storage"], label="Rule")
    ax2.plot(bundle.series["DP"]["storage"], label="DP")
    ax2.plot(bundle.series["HOSM+FWCR+MPC"]["storage"], label="HOSM+FWCR+MPC", linewidth=2)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Storage (m³)")
    ax2.legend(ncol=3)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    outdir = Path("experiments/results")
    outdir.mkdir(parents=True, exist_ok=True)
    bundle.data.to_csv(outdir / "app_generated_runoff.csv", index=False)
    bundle.metrics.to_csv(outdir / "app_metrics.csv", index=False)
    st.download_button("下载当前模拟数据 CSV", data=bundle.data.to_csv(index=False), file_name="simulated_runoff.csv")
    st.download_button("下载当前指标 CSV", data=bundle.metrics.to_csv(index=False), file_name="benchmark_metrics.csv")
else:
    st.write("👈 请在左侧设置参数并点击按钮运行。")
