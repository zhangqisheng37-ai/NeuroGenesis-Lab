# config — 配置系统

集中配置管理模块，提供仿真运行所需的全部参数定义。

## 核心类

- **`SimConfig`** — 顶层仿真配置，包含 `NeuronParams`、`NetworkParams`、`EvolutionParams`
- **`NeuronParams`** — 神经元动力学参数（v_rest, v_th, tau_m 等）
- **`NetworkParams`** — 网络级参数（繁殖阈值、能量衰减、自适应机制等）
- **`EvolutionParams`** — 演化参数（适应度权重、变异幅度、拥塞惩罚等）
- **`Config`** — 旧版简化配置（向后兼容）

## 工具函数

- `default_config()` — 返回旧版默认 Config
- `default_sim_config()` — 返回默认 SimConfig
- `seed_all(seed)` — 统一设置 random + numpy 种子

## 共享常量

模块导出 `SCHEMA_VERSION`、自适应阈值常量（`DENSITY_HIGH_THRESHOLD` 等）、遗传属性列表（`HERITABLE_ATTRS`）等。

::: neurogenesis_lab.config
