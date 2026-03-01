# 风险清单（文档与实现一致性视角）

> 更新日期：2026-03-01
> 代码基线：v0.6.0 / Schema 2.0.0

本清单聚焦“文档与当前实现对齐”风险，而非功能设计路线图。

## 高优先级

### R1：版本/Schema 漂移

- 风险：文档仍引用旧版本或旧 schema，导致实验复现失败。
- 现状：已统一到 `v0.6.0` 与 `2.0.0`。
- 建议：每次版本变更同时更新 `README`、`docs/CHANGELOG.md`、`docs/parameters.md`。

### R2：CLI 参数口径不一致

- 风险：文档示例命令不可解析，用户误判为代码故障。
- 现状：已以 `cli/parser.py` 重建参数表。
- 建议：以 `python -m neurogenesis_lab run --help` 为回归基线。

### R3：行为叙事滞后

- 风险：文档描述与真实运行路径不一致（如 run 实际使用的引擎、checkpoint 语义）。
- 现状：已按 `cli/run.py` 同步。
- 建议：功能说明必须链接到消费代码位置。

## 中优先级

### R4：指标字段解释偏差

- 风险：分析脚本使用错误字段语义，导致结论偏差。
- 现状：已按 `protocols.py` + `observability.py` 同步。
- 建议：所有指标文档优先使用 `TickSnapshot` / `EngineSummary` 字段名。

### R5：实验配置导航缺失

- 风险：用户不知道 `experiments/*.toml` 的用途与入口。
- 现状：已在 README/quickstart 增加实验索引。
- 建议：新增配置文件时同步更新索引区。

## 低优先级

### R6：历史阶段文档与主文档冲突

- 风险：阶段清单中的历史叙事被误当作当前事实。
- 建议：阶段文档保留历史属性，不作为参数真值来源。

## 建议的持续验证（每次文档更新后执行）

```bash
python -m neurogenesis_lab --help
python -m neurogenesis_lab run --help
python -m neurogenesis_lab benchmark --help
python -m neurogenesis_lab info
python -m neurogenesis_lab run --ticks 5 --seed 42
python -m neurogenesis_lab benchmark --neurons 1000 --ticks 10 --rust-size 5000
```

并进行全仓扫描：

- 版本号是否残留旧值
- Schema 是否残留旧值
- 参数名是否存在已删除/拼写错误
