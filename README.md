# 基于分层优化与MPC的实时水库调度实验框架

该仓库实现了一个可复现实验框架，包含：
1. GPR短期径流预测（含逐步回归特征筛选）
2. 分层优化调度（上层DE优化 + 下层闸门分配）
3. 流量分级水位控制规则（FWCR）
4. MPC滚动优化与反馈校正
5. 基线方法（规则法、离散DP）
6. PPO强化学习训练脚本（用于基准增强）
7. 可视化演示应用（支持点击生成模拟数据、运行算法与对比基线）

## 快速开始
```bash
python -m pip install -e .
python scripts/generate_data.py --n_steps 2880 --scenario mixed
python scripts/run_experiments.py --data experiments/runoff.csv
python scripts/run_sensitivity.py
python scripts/train_rl.py
```

## 可视化展示应用（产品化入口）
方式1：
```bash
streamlit run app.py
```
方式2（Python启动入口）：
```bash
python scripts/launch_dashboard.py
```

控制台覆盖流程：
- 数据生成（场景/随机种子）
- 算法训练（GPR自动训练 + PPO手动训练入口）
- 算法效果可视化（来水-下泄/库容）
- 性能增长对比（相对Rule基线）

## 测试审查建议（正确性与易用性）
```bash
python -m compileall src scripts app.py tests
PYTHONPATH=src pytest -q
python scripts/run_experiments.py --data experiments/runoff.csv
python scripts/run_sensitivity.py --data experiments/runoff.csv
```
建议重点关注：
- `metrics.csv` 中 `peak_shaving`、`energy`、`runtime_s` 的相对排序是否符合预期
- `release_compare.png` 是否出现明显异常（剧烈振荡、物理越界）
- 不同 `FWCR` 阈值与预测时域下结果是否呈现稳定趋势

实验输出位于 `experiments/results/`。
