# 更新日志（Changelog）

本文档记录重要版本变更。参数与行为真值以代码为准：

- `src/neurogenesis_lab/cli/parser.py`
- `src/neurogenesis_lab/cli/run.py`
- `src/neurogenesis_lab/config.py`

---

## 未发布（文档同步批次，2026-03-01）

### 文档与代码事实对齐

- 全量校对 `README.md`、`docs/parameters.md`、`docs/quickstart.md`。
- 统一 `run / benchmark / info` 参数口径（名称、默认值、帮助语义）。
- 明确输出治理能力：`--log-interval`、`--gzip-output`、`--max-file-size`。
- 明确异步日志能力：`--async-log`、`--log-buffer-size`。
- 明确 checkpoint 运行语义：`--checkpoint-interval`、`--checkpoint-dir`、`--resume-from`。
- 同步双引擎协议叙事：`SimEngine` / `TickSnapshot` / `EngineSummary`。
- 更新教程/案例命令，确保在当前 CLI 下可解析。

---

## v0.6.0

### 已落地能力（当前主线）

- 版本号：`0.6.0`（`src/neurogenesis_lab/__init__.py`）。
- Schema：`2.0.0`（`src/neurogenesis_lab/config.py`）。
- CLI 子包化：`cli/parser.py`、`cli/run.py`、`cli/benchmark.py`、`cli/info.py`、`cli/config_builder.py`。
- `Network` 与 `VectorizedNetwork` 统一到 `SimEngine` 协议。
- `VectorizedNetwork` 支持多模型向量化动力学（LIF / Izhikevich / AdEx）。
- 观测链路支持异步双缓冲写盘与输出轮转。

---

## v0.4.x（历史）

- 引入可视化与长程稳定机制。
- 引入 checkpoint 保存与恢复。
- 引入 Rust 桥接层与微基准路径。

> 说明：历史阶段文档仅用于追溯，不作为当前参数真值来源。
