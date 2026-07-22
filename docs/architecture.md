# 系统架构（Architecture）

本文档描述当前代码基线（v0.6.1）下的真实架构。

## 1. 总览

NeuroGenesis Lab 采用「CLI + 配置 + 双引擎 + 可观测性」分层：

1. CLI 解析参数并构建 `SimConfig`
2. `run` 命令初始化 `Network` 并驱动 tick 循环
3. `Network` / `VectorizedNetwork` 通过 `SimEngine` 协议统一接口
4. `NetworkObserver` + `ObservabilityLogger` 负责 JSONL 观测输出
5. 双引擎共享 `adaptive.py` 的自适应机制（衰减/繁殖成本动态调节）
6. `_components.py` 提供 `ComponentMixin` 混入基类，`Neuron` 和 `Network` 通过委托模式将字段透明代理到子组件

## 2. 关键模块

```text
src/neurogenesis_lab/
├── __init__.py                 # __version__ = 0.6.1
├── __main__.py                 # 向后兼容入口，转发到 cli.main
├── _components.py              # ComponentMixin — 组件委托混入基类
├── adaptive.py                 # 自适应衰减/繁殖 — 双引擎共享逻辑
├── config.py                   # SCHEMA_VERSION=2.0.0, SimConfig/Params
├── core_bridge.py              # Rust 桥接与 Python fallback
├── encoding.py                 # 编码系统（Event/Tokenizer/Encoder）
├── evolution.py                # 引擎无关的演化决策函数
├── logger.py
├── network/                    # 标准引擎子包（继承 ComponentMixin）
│   ├── __init__.py
│   ├── engine.py               # Network 主类（组合 dynamics/lifecycle/persistence Mixin）
│   ├── event_queue.py          # EventQueue — 事件优先队列
│   ├── corpse.py               # Corpse / CorpsePool 尸体回收
│   ├── state.py                # NetworkConfig / NetworkState 组件
│   ├── dynamics.py             # 动力学 Mixin
│   ├── lifecycle.py            # 生命周期 Mixin
│   └── persistence.py          # checkpoint 保存与恢复
├── vectorized/                 # 向量化引擎子包
│   ├── __init__.py
│   ├── engine.py               # VectorizedNetwork 主类
│   ├── spatial.py              # SpatialGrid — 空间索引
│   ├── state.py                # VectorizedState — 全字段容器
│   ├── dynamics.py             # 多模型分组动力学 Mixin
│   ├── propagation.py          # 批量稀疏传播 Mixin
│   ├── lifecycle.py            # 生命周期 Mixin
│   └── persistence.py          # checkpoint 持久化
├── learning/                   # 学习引擎子包
│   ├── __init__.py
│   ├── auto.py                 # AutoLearner 自动学习包装器
│   ├── expression.py           # 表达式解析/模式挖掘/查询引擎
│   └── rules.py                # RuleStore / RuleDiscovery / RuleComposer
├── neuron.py                   # LIF/Izhikevich/AdEx 神经元（继承 ComponentMixin）
├── observability.py            # 观测采集 + 写盘
├── protocols.py                # TickSnapshot / EngineSummary / SimEngine
├── visualization.py            # 可视化与实时监控
└── cli/                        # CLI 子包
    ├── __init__.py
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

`src/neurogenesis_lab/network/` 子包，主类位于 `engine.py`：

- 继承 `ComponentMixin`，配置/状态字段透明代理到 `NetworkConfig` / `NetworkState` 子组件
- 事件队列驱动（`ready_events`）
- 演化：繁殖、死亡、尸体回收
- 自适应机制：通过 `adaptive.calculate_adaptive_decay()` / `calculate_adaptive_reproduction_share()` 共享实现
- 摘要额外字段：`corpse_pool_free`、`corpse_pool_reused`

### 4.2 VectorizedNetwork（向量化引擎）

`src/neurogenesis_lab/vectorized/` 子包，主类位于 `engine.py`：

- NumPy + SciPy 稀疏矩阵路径
- 多模型分组动力学（LIF/Izhikevich/AdEx）
- 批量稀疏传播：`_propagate_spikes` 使用稀疏切片 + 按延迟分组调度，消除逐神经元 getrow 开销
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

- 版本：`0.6.1`（`src/neurogenesis_lab/__init__.py`）
- Schema：`2.0.0`（`src/neurogenesis_lab/config.py`）

## 7. 说明

- 当前 `run` 命令路径使用 `Network`（标准引擎）。
- `benchmark` 同时执行 `VectorizedNetwork`、`Network` 与 Rust 并行 LIF 微基准。
- 架构叙事以代码实现为准，不引入无代码证据的设计假设。
