# adaptive — 自适应机制

双引擎共享的自适应衰减与繁殖成本动态调节逻辑。

## 核心函数

- **`calculate_adaptive_decay(enabled, base_decay, alive_count, max_neurons)`**  
  根据种群密度计算自适应能量衰减系数。密度高于 `DENSITY_HIGH_THRESHOLD` 时增加衰减，低于 `DENSITY_LOW_THRESHOLD` 时减少衰减。

- **`calculate_adaptive_reproduction_share(enabled, base_share, alive_count, max_neurons)`**  
  根据种群规模计算自适应繁殖能量分享比例。接近容量上限时增加成本，种群稀少时降低成本以鼓励繁殖。

::: neurogenesis_lab.adaptive
