# 问题四代码导航

本目录保留完整复现链。**普通阅读不需要逐个打开所有 CSV/JSON**；先看根目录 `../README.md` 和 `../output/`。

## 1. 核心代码按执行链分类

| 阶段 | 文件 | 作用 |
|---|---|---|
| 数据与时间 | `q4_data.py` | 官方附件读取、自然日/计划日双时间坐标 |
| 信息边界 | `q4_information.py` | 决策时刻可用信息、42天历史窗 |
| 预测 | `q4_forecast.py`、`q4_price.py` | 负荷/PV权限、波动电价因果预测 |
| 场景 | `q4_scenarios.py`、`q4_tree.py` | 同源残差路径、场景约简与事件树 |
| 物理 | `q4_flow.py`、`q4_control.py` | 唯一源—汇物理核、10分钟执行 |
| 结算 | `q4_settlement.py` | 原始合同 B 锚定结算、5c紧急购电 |
| 优化 | `q4_opt.py` | 事件层 LP / SAA / DRO / 字典序 |
| 回放 | `q4_sim.py`、`q4_metrics.py` | 连续 SOC 回放、正式指标 |
| 基线/冻结 | `q4_bridge.py`、`q4_select.py` | R0/C0桥接、1月复杂度冻结 |
| 正式期 | `q4_formal.py` | 334天冻结策略正式回放 |
| Oracle | `q4_oracle.py`、`q4_oracle_run.py` | Price Oracle 与 Full-information Oracle |
| 导出 | `q4_export.py` | 官方 Excel 与指定日期表 |
| 图表 | `q4_figures.py` | 正文图与附图 |
| 论文材料 | `q4_paper.py` | 生成 `../问题4论文材料.md` |
| 验收 | `q4_validate.py` | T01—T42 独立核验 |

## 2. 消融脚本

`q4_ablations*.py`、`q4_e0_finish.py`、`q4_e13_finish.py` 等用于冻结后的 E0—E15 验证。最终统一结果见：

- `ablations/ablation_matrix_E0_E15.json`
- `../output/09_鲁棒性与消融验证.md`

`q4_settlement_reopt.py` 属于结算语义实验代码；正式材料对 E10 **只保留固定主策略 exposure**，不把未完全可靠的替代结算重优化结果用于排名或冻结。

## 3. 机器产物目录

- `q4_2/`：Q4-2 正式 334 天账本与 checkpoint
- `q4_3/`：Q4-3 正式 334 天账本与 checkpoint
- `ablations/`：E0—E15 原始机器证据
- `../output/02_model_selection/`：1月模型选择全部候选

其中 `q4_2/physical_10min.csv`、`q4_2/contract_ledger.csv`、`q4_2/event_ledger.csv`、`q4_2/daily_ledger.csv` 与 `q4_3/` 下对应文件是两分支权威回放账本；`q4_2/checkpoints/`、`q4_3/checkpoints/` 只是分段运行恢复与复核材料，不是论文阅读入口。

## 4. 冻结配置与关键清单

- `config_frozen.yaml`：正式冻结配置
- `january_frozen_selection.json`：1月复杂度冻结结果
- `oracle_audit.json`：Oracle边界
- `export_manifest.json`：Excel/指定日期输出哈希
- `validation.json`：T01—T42逐项证据
- `final_acceptance.json`：最终验收状态

## 5. 推荐复核顺序

不要从 checkpoint 开始。人工复核建议：

`config_frozen.yaml` → `january_frozen_selection.json` → `q4_2/formal_summary.json`、`q4_3/formal_summary.json` → `oracle_audit.json` → `ablations/ablation_matrix_E0_E15.json` → `validation.json`。
