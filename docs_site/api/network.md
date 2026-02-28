# network — 标准网络引擎

基于事件队列的标准仿真引擎，实现 `SimEngine` 协议。

## 核心类

- **`Network`** — 主引擎类，内部委托 `NetworkConfig` + `NetworkState` 组件
- **`NetworkConfig`** — 纯配置字段（22 个参数）
- **`NetworkState`** — 运行时统计状态
- **`EventQueue`** — 优先队列（heapq）管理事件分发
- **`Corpse`** — 尸体数据类（位置、能量、TTL）
- **`CorpsePool`** — 尸体对象池，减少 GC 抖动

## 关键方法

- `tick(tick, dt, run_evolution)` — 执行一步仿真，返回 TickSnapshot
- `connect(source_id, target_id, weight, delay)` — 建立连接
- `connect_by_proximity(radius)` — 空间距离自动连接
- `check_reproduction()` / `check_death()` — 演化步骤
- `save_checkpoint(path, tick)` / `load_checkpoint(path)` — 持久化

::: neurogenesis_lab.network
