# 系统架构（Architecture）

本文档描述当前代码基线（v0.6.0）下的真实架构。

## 1. 总览

NeuroGenesis Lab 采用「CLI + 配置 + 双引擎 + 可观测性」分层：

1. CLI 解析参数并构建 `SimConfig`
2. `run` 命令初始化 `Network` 并驱动 tick 循环
3. `Network` / `VectorizedNetwork` 通过 `SimEngine` 协议统一接口
4. `NetworkObserver` + `ObservabilityLogger` 负责 JSONL 观测输出

## 2. 关键模块

```text
src/neurogenesis_lab/
├── __init__.py                 # __version__ = 0.6.0
├── __main__.py                 # 向后兼容入口，转发到 cli.main
├── config.py                   # SCHEMA_VERSION=2.0.0, SimConfig/Params
├── protocols.py                # TickSnapshot / EngineSummary / SimEngine
├── neuron.py                   # LIF/Izhikevich/AdEx 神经元
├── network.py                  # 标准引擎（run 命令当前默认）
├── vectorized.py               # 向量化引擎（benchmark 主要使用）
├── learning.py                 # AutoLearner / RuleDiscovery
├── observability.py            # 观测采集 + 写盘
├── core_bridge.py              # Rust 桥接与 fallback
└── cli/
    ├── parser.py               # CLI 参数定义
    ├── config_builder.py       # TOML + CLI -> SimConfig
    ├── run.py                  # run 子命令
    ├── benchmark.py            # benchmark 子命令
    └── info.py                 # info 子命令
```

## 3. CLI 运行路径（run）

`src/neurogenesis_lab/cli/run.py` 中，`cmd_run()` 流程如下：

1. `build_sim_config(args)` 构建配置
2. `validate_run_config(cfg)` 校验
3. `seed_all(cfg.seed)` 统一随机种子
4. `_init_network()`：新建或 `--resume-from` 恢复 `Network`
5. `_init_logging()`：初始化 logger / observer / 可视化
6. `_run_simulation_loop()`：
   - `network.tick(t, dt=cfg.dt)`
   - `AutoLearner.on_tick(...)`
   - `logger.log_tick(...)`
   - checkpoint 条件保存
7. `_finalize_run()` 写 summary

## 4. 双引擎与统一协议

统一协议定义于 `src/neurogenesis_lab/protocols.py`：

- `TickSnapshot`（每 tick 快照）
- `EngineSummary`（运行摘要）
- `SimEngine`（协议，约束 `tick()` + `get_summary()`）

### 4.1 Network（标准引擎）

`src/neurogenesis_lab/network.py`：

- 事件队列驱动（`ready_events`）
- 演化：繁殖、死亡、尸体回收
- 自适应机制：`adaptive_decay` / `adaptive_reproduction`
- 摘要额外字段：`corpse_pool_free`、`corpse_pool_reused`

### 4.2 VectorizedNetwork（向量化引擎）

`src/neurogenesis_lab/vectorized.py`：

- NumPy + SciPy 稀疏矩阵路径
- 多模型分组动力学（LIF/Izhikevich/AdEx）
- 同样返回 `TickSnapshot` / `EngineSummary`
- 支持独立 checkpoint（`save_checkpoint` / `load_checkpoint`）

## 5. 可观测性架构

`src/neurogenesis_lab/observability.py`：

- `NetworkObserver`：采集 `metrics` / `learning` / `trends`
- `ObservabilityLogger`：写入 `config` / tick 报告 / `summary`
- `JsonlWriter`：同步写盘
- `AsyncJsonlWriter`：异步双缓冲写盘
- `create_experiment_logger()`：自动生成实验目录

输出治理能力由 run CLI 参数启用：

- 采样：`--log-interval`
- 压缩：`--gzip-output`
- 轮转：`--max-file-size`
- 异步：`--async-log` + `--log-buffer-size`

## 6. 版本与 Schema

- 版本：`0.6.0`（`src/neurogenesis_lab/__init__.py`）
- Schema：`2.0.0`（`src/neurogenesis_lab/config.py`）

## 7. 说明

- 当前 `run` 命令路径使用 `Network`（标准引擎）。
- `benchmark` 同时执行 `VectorizedNetwork`、`Network` 与 Rust 并行 LIF 微基准。
- 架构叙事以代码实现为准，不引入无代码证据的设计假设。
