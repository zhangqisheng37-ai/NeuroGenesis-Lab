# vectorized — 向量化引擎

基于 NumPy + scipy.sparse 的高性能仿真引擎，支持万级神经元规模。实现 `SimEngine` 协议。

## 核心类

- **`VectorizedNetwork`** — 向量化主引擎
  - 多模型动力学（LIF/Izhikevich/AdEx，通过 `model_type` 整数向量区分）
  - 尸体回收系统（`corpse_mask` / `corpse_timer` / `corpse_energy`）
  - 自适应衰减/繁殖 + 激危保护模式
  - checkpoint 持久化
- **`SpatialGrid`** — 空间网格索引，加速近邻查找

## 关键方法

- `init_neurons(count, ...)` — 初始化神经元数组
- `connect_radius(radius, weight, delay)` — 空间半径连接
- `connect_knn(k, weight, delay)` — k 近邻连接
- `tick(tick, dt, input_currents)` — 向量化一步
- `save_checkpoint(path, tick)` / `load_checkpoint(path)` — 持久化

::: neurogenesis_lab.vectorized
