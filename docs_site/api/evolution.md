# evolution — 演化策略

引擎无关的纯决策函数模块，集中管理 `Network` 与 `VectorizedNetwork` 共享的演化逻辑。

## 核心函数

- **`update_endangered_mode(current_mode, alive_count, threshold)`** — 根据存活数更新濒危模式标志
- **`select_rescued(candidates, deficit)`** — 从死亡候选中挑选待复活者（Network 标量路径）
- **`select_rescued_indices(energies, dead_indices, deficit)`** — 从死亡索引中挑选待复活者（Vectorized 数组路径）
- **`can_reproduce(energy, threshold)`** — 判定单个神经元是否达到繁殖能量阈值
- **`reproduction_mask(alive, energy, threshold)`** — 计算可繁殖掩码（Vectorized 数组路径）
- **`child_energy(parent_energy, share)`** — 按分享比例派生子代初始能量
- **`decay_corpse_energy(energy, rate)`** — 尸体能量自然衰减
- **`is_corpse_expired(ttl, energy)`** — 判定尸体是否应移除

::: neurogenesis_lab.evolution
