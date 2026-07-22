# 更新日志（Changelog）

本文档记录重要版本变更。参数与行为真值以代码为准：

- `src/neurogenesis_lab/cli/parser.py`
- `src/neurogenesis_lab/cli/run.py`
- `src/neurogenesis_lab/config.py`

---

## v0.6.1（2026-07-22）

### 代码质量与性能优化

- **自适应逻辑提取**：新增 `adaptive.py` 模块，将 `Network` 和 `VectorizedNetwork` 中重复的 `_calculate_adaptive_decay` / `_calculate_adaptive_reproduction_share` 抽取为 `calculate_adaptive_decay()` / `calculate_adaptive_reproduction_share()` 共享函数，消除 88 行重复代码。
- **组件委托 Mixin**：新增 `_components.py` 模块，提供 `ComponentMixin` 混入基类。`Neuron` 和 `Network` 的 `__getattr__` / `__setattr__` / `__post_init__` / `__setstate__` 委托逻辑统一由此基类管理，净减少 ~22 行同时消除两处重复实现。
- **向量化传播优化**：`VectorizedNetwork._propagate_spikes` 重写为批量稀疏矩阵切片 + 按延迟分组调度，将逐 target 的 `np.zeros(n)` 分配次数从 O(边数) 降为 O(不同延迟数)，`_schedule_currents` 调用次数大幅减少。

### 大模块按功能拆分（子包化）

- **`network.py`（~900 行）→ `network/` 子包**：`EventQueue`（`event_queue.py`）、`Corpse`/`CorpsePool`（`corpse.py`）、`NetworkConfig`/`NetworkState`（`state.py`）拆为独立模块；`Network` 主类经 dynamics/lifecycle/persistence 三个 Mixin 组合（`engine.py`），配置/状态字段仍由 `ComponentMixin` 透明代理。
- **`vectorized.py`（~1000 行）→ `vectorized/` 子包**：`SpatialGrid`（`spatial.py`）独立；全部字段集中于 `VectorizedState`（`state.py`），`VectorizedNetwork` 经 dynamics/propagation/lifecycle/persistence 四个 Mixin 组合（`engine.py`），字段经多重继承单一来源、零重复声明。
- **`learning.py`（~850 行）→ `learning/` 子包**：按类拆为 `expression.py`（`MemoryBank`/`PatternMiner`/`QueryEngine`）、`rules.py`（`RuleStore`/`RuleDiscovery`/`RuleComposer`）、`auto.py`（`AutoLearner`）。
- **公共 API 完全不变**：各子包 `__init__.py` 原样 re-export 全部符号（含 `_parse_expression`、`mine_pattern_counts_safe`），`from neurogenesis_lab.network import Network` 等导入路径无回归。

### 性能回归测试

- 新增 `tests/test_performance_regression.py`：以**确定性操作计数断言**守护向量化优化，锁定结构性不变量（`_propagate_spikes` 按投递 tick 分组调度、每 tick 每种模型 step 内核仅调用一次），不依赖挂钟阈值，CI 稳定；后续重构若使优化退化将立即失败。


### 工程改进

- 修复 `test_cli_config.py` 导入路径（`__main__._build_sim_config` → `cli.config_builder.build_sim_config`）。
- 移除 `pyproject.toml` 中的 `--basetemp=.pytest_tmp`，pytest 改用系统临时目录避免 Windows 权限问题。
- mypy 零错误、ruff 零警告，所有 177+ 核心测试通过。

---

## v0.6.0（2026-03-05）

### 工程基础设施完善（Phase 7 完成）

#### CI/CD 管道
- 新增 `.github/workflows/python-test.yml`：多平台（Ubuntu/Windows）× 多版本（3.10/3.11/3.12）矩阵测试，覆盖率≥90%。
- 新增 `.github/workflows/lint.yml`：ruff check + format + mypy 类型检查。
- 新增 `.github/workflows/coverage.yml`：PR 覆盖率报告自动评论。
- 新增 `.github/workflows/docs.yml`：自动构建并部署 API 文档到 GitHub Pages。
- 新增 `.github/BRANCH_PROTECTION_GUIDE.md`：分支保护规则配置指南。

#### 双引擎功能对齐
- `VectorizedNetwork` 新增尸体回收系统（`corpse_mask`、`corpse_timer`、空间索引加速）。
- `VectorizedNetwork` 新增自适应衰减与繁殖机制（动态调整 `energy_decay` 和 `reproduction_share`）。
- `VectorizedNetwork` 新增多模型支持（LIF + Izhikevich + AdEx 向量化）。
- `VectorizedNetwork` 新增学习集成（`AutoLearner` + 预测奖励机制）。
- `VectorizedNetwork` 新增 checkpoint 持久化（`save_checkpoint` / `load_checkpoint`）。

#### CLI 模块化
- 将 `__main__.py`（732 行）拆分为 `cli/` 子包：
  - `cli/parser.py`：参数定义
  - `cli/config_builder.py`：配置构建与 TOML 解析
  - `cli/run.py`：仿真运行逻辑
  - `cli/benchmark.py`：性能基准
  - `cli/info.py`：版本信息
- `__main__.py` 精简为入口存根（<10 行）。

#### API 文档自动生成
- 集成 mkdocs-material + mkdocstrings-python。
- 自动生成 9 个核心模块的 API 文档。
- 整合教程、案例研究到统一导航。
- GitHub Actions 自动部署到 GitHub Pages。

#### 测试基础设施
- 新增 `tests/conftest.py`：共享 fixture（`sample_config`、`small_network`、`small_vectorized`、`tmp_output_dir`）。
- 新增 30+ 测试用例，覆盖率从 90% 提升至 92.61%。
- 新增端到端测试：run → checkpoint → resume → 结果对比。

#### 类型标注收紧
- 修复 `vectorized.py`、`visualization.py`、`learning.py` 的弱类型标注。
- 启用 mypy 严格模式（`disallow_untyped_defs`、`warn_return_any`）。
- 集成 mypy 到 lint CI 工作流。

#### 项目发布准备
- 新增 `LICENSE`（MIT）。
- 新增 `CONTRIBUTING.md`：开发环境、分支策略、提交规范。
- 新增 `src/neurogenesis_lab/py.typed`（PEP 561 类型标记）。
- README 新增徽章（CI 状态、覆盖率、Python 版本、License）。

#### 文档同步
- 全量校对 `README.md`、`docs/parameters.md`、`docs/quickstart.md`。
- 统一 `run / benchmark / info` 参数口径。
- 更新架构文档、风险清单至 v0.6.x 口径。
- 开发清单 07 标记为已完成。

---

## v0.5.0（历史）

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
