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

## 可视化展示应用
```bash
streamlit run app.py
```
在页面侧边栏可调整：
- 模拟时长、随机种子、水文情景（wet/dry/extreme 等）
- FWCR阈值
- MPC预测时域与决策步长

点击“生成模拟数据并运行对比实验”后，系统会自动：
- 生成模拟来水
- 训练GPR并运行HOSM+FWCR+MPC
- 与Rule/DP基线对比
- 输出表格指标与曲线图，并可下载CSV

实验输出位于 `experiments/results/`。
