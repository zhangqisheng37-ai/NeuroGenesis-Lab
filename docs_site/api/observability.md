# observability — 可观测性

全链路数据观测与日志系统。

## 核心类

- **`JsonlWriter`** — 同步 JSONL 写入器
- **`AsyncJsonlWriter`** — 双缓冲异步 JSONL 写入器
- **`NetworkObserver`** — 网络状态采集器（指标、趋势、调试信息）
- **`ObservabilityLogger`** — 实验日志管理器（采样、轮转、压缩）

## 工具函数

- `create_experiment_logger(...)` — 创建带实验 ID 的日志器

::: neurogenesis_lab.observability
