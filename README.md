# 基于分层优化与MPC的实时水库调度实验框架

该仓库实现了一个可复现实验框架，包含：
1. GPR短期径流预测（含逐步回归特征筛选）
2. 分层优化调度（上层DE优化 + 下层闸门分配）
3. 流量分级水位控制规则（FWCR）
4. MPC滚动优化与反馈校正
5. 基线方法（规则法、离散DP）
6. PPO强化学习训练脚本（用于基准增强）
7. 可视化演示应用（支持点击生成模拟数据、运行算法与对比基线）
8. 水文专家导向的概念性降雨-径流模拟器（快流+基流+汇流）

## 快速开始
```bash
python -m pip install -e .
python scripts/generate_data.py --n_steps 2880 --scenario mixed
python scripts/run_experiments.py --data experiments/runoff.csv
python scripts/run_sensitivity.py
python scripts/train_rl.py
```

## 水文模拟器（推荐用于算法管理效果模拟）
```bash
python scripts/simulate_hydrology.py --n_steps 2880 --scenario mixed --out experiments/hydro_sim.csv
python scripts/run_experiments.py --data experiments/hydro_sim.csv
```
说明：
- `simulate_hydrology.py` 会输出包含 `precip_mm_h / et0_mm_h / soil_mm / quickflow / baseflow / qin` 的时序数据。
- 可用于更“物理过程友好”的调度算法验证，而非仅依赖统计合成流量。
- 在可视化面板中可选择 `hydro_physics` 场景，一键执行管理效果对比。

## 可视化展示应用（产品化入口）
方式1：
```bash
streamlit run app.py
```
方式2（Python启动入口）：
```bash
python scripts/launch_dashboard.py
```

## 产品可销售审查（PM + 架构视角）
先执行环境与可销售就绪度检查：
```bash
python scripts/product_readiness_audit.py
```
可视化面板新增能力：
- 演示预设（平衡/防洪优先/物理过程）
- 失败可读错误提示
- 可销售就绪度页签（依赖/能力检查）
- 一键导出业务汇报数据（CSV）

当前距离“可销售”的关键差距（建议）：
- 补齐真实流域参数化与校准工具
- 提供多项目租户管理、权限、审计日志
- 提供API服务化部署与SLA监控
- 增加报告模板（PDF/Word）与自动化回归基准

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

## 上线前测试建议（资深测试开发）
```bash
python scripts/pre_release_check.py
# 严格模式（任一项失败即退出非0）
python scripts/pre_release_check.py --strict
```
该脚本会执行：
- 代码编译检查
- 产品就绪度审计
- pytest回归（在依赖不足时给出WARN）
