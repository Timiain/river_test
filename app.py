from __future__ import annotations

from pathlib import Path

import streamlit as st

from reservoir.dashboard import (
    PanelConfig,
    build_growth_table,
    render_release_figure,
    render_storage_figure,
    run_panel_experiment,
    run_rl_training,
)


st.set_page_config(page_title="Reservoir Decision Intelligence Panel", layout="wide", initial_sidebar_state="expanded")
st.title("🌊 水库智能调度可视化控制台")
st.caption("产品视角覆盖全流程：数据生成 → 算法训练 → 策略评估 → 性能增长对比。")

if "bundle" not in st.session_state:
    st.session_state.bundle = None

with st.sidebar:
    st.header("⚙️ 场景与算法参数")
    n_steps = st.slider("模拟总时长（小时）", min_value=240, max_value=24 * 240, value=24 * 90, step=24)
    seed = st.number_input("随机种子", min_value=0, value=42, step=1)
    scenario = st.selectbox("水文场景", ["mixed", "wet", "dry", "extreme", "climate_trend", "hydro_physics"])
    fwcr_threshold = st.slider("FWCR阈值 (m³/s)", min_value=8000, max_value=20000, value=14000, step=500)
    pred_h = st.select_slider("MPC预测时域 (h)", options=[24, 36, 48], value=36)
    step_h = st.select_slider("MPC决策步长 (h)", options=[1, 2, 3], value=2)

    st.divider()
    run_btn = st.button("🚀 一键运行完整评估", type="primary", use_container_width=True)

if run_btn:
    cfg = PanelConfig(
        n_steps=int(n_steps),
        seed=int(seed),
        scenario=scenario,
        fwcr_threshold=float(fwcr_threshold),
        pred_h=int(pred_h),
        step_h=int(step_h),
    )
    with st.spinner("正在执行：生成数据 + 训练预测 + 三方法评估..."):
        st.session_state.bundle = run_panel_experiment(cfg)
    st.success("评估完成。")

bundle = st.session_state.bundle

if bundle is None:
    st.info("请先点击左侧 **一键运行完整评估**。")
else:
    metrics = bundle.metrics.copy()
    growth = build_growth_table(metrics)

    row = growth.set_index("method")
    best_peak_method = growth.sort_values("peak_shaving", ascending=False).iloc[0]["method"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最佳削峰方法", best_peak_method)
    c2.metric("HOSM削峰率", f"{row.loc['HOSM+FWCR+MPC', 'peak_shaving']:.4f}")
    c3.metric("HOSM相对Rule发电增长", f"{row.loc['HOSM+FWCR+MPC', 'energy_growth_vs_rule_%']:.2f}%")
    c4.metric("HOSM相对Rule削峰增长", f"{row.loc['HOSM+FWCR+MPC', 'peak_growth_vs_rule_%']:.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 流程覆盖检查", "📈 结果可视化", "🧪 性能增长对比", "🤖 训练入口"])

    with tab1:
        st.subheader("产品经理视角：操作覆盖性")
        st.markdown(
            """
- ✅ 数据生成：支持场景选择 + 随机种子控制，保证复现实验。
- ✅ 算法训练：自动执行GPR预测训练，支持独立触发PPO训练。
- ✅ 算法评估：Rule / DP / HOSM+FWCR+MPC统一对比。
- ✅ 效果展示：流量、库容、指标表、性能增长率完整输出。
- ✅ 导出能力：可下载模拟数据与指标CSV用于汇报。
            """
        )
        st.dataframe(growth, use_container_width=True)

    with tab2:
        st.subheader("来水与下泄过程")
        st.pyplot(render_release_figure(bundle))
        st.subheader("库容轨迹")
        st.pyplot(render_storage_figure(bundle))

    with tab3:
        st.subheader("性能增长对比（相对 Rule 基线）")
        show_cols = [
            "method",
            "peak_shaving",
            "energy",
            "runtime_s",
            "peak_growth_vs_rule_%",
            "energy_growth_vs_rule_%",
        ]
        st.dataframe(growth[show_cols], use_container_width=True)
        st.bar_chart(growth.set_index("method")[["peak_growth_vs_rule_%", "energy_growth_vs_rule_%"]])

    with tab4:
        st.subheader("强化学习训练入口（可选）")
        rl_epochs = st.slider("PPO训练轮数", min_value=5, max_value=100, value=20, step=5)
        rl_h = st.slider("单次训练时域", min_value=24, max_value=120, value=72, step=12)
        if st.button("开始训练PPO并保存模型"):
            with st.spinner("训练中..."):
                model_path = run_rl_training(seed=int(seed), epochs=int(rl_epochs), horizon=int(rl_h))
            st.success(f"已保存模型：{model_path}")

    outdir = Path("experiments/results")
    outdir.mkdir(parents=True, exist_ok=True)
    bundle.data.to_csv(outdir / "app_generated_runoff.csv", index=False)
    growth.to_csv(outdir / "app_metrics_growth.csv", index=False)

    st.download_button("下载模拟数据 CSV", data=bundle.data.to_csv(index=False), file_name="simulated_runoff.csv")
    st.download_button("下载指标与增长对比 CSV", data=growth.to_csv(index=False), file_name="benchmark_metrics_growth.csv")
