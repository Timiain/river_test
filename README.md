# 基于分层优化与MPC的实时水库调度实验框架

该仓库实现了一个可复现实验框架，包含：
1. GPR短期径流预测（含逐步回归特征筛选）
2. 分层优化调度（上层DE优化 + 下层闸门分配）
3. 流量分级水位控制规则（FWCR）
4. MPC滚动优化与反馈校正
5. 基线方法（规则法、离散DP）
6. PPO强化学习训练脚本（用于基准增强）

## 快速开始
```bash
python -m pip install -e .
python scripts/generate_data.py --n_steps 2880
python scripts/run_experiments.py
python scripts/run_sensitivity.py
python scripts/train_rl.py
```

实验输出位于 `experiments/results/`。
