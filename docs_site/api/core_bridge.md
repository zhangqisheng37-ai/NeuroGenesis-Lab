# core_bridge — Rust 桥接层

Rust 核心扩展（`neurogenesis_core`）的 Python 桥接层，提供可用性探测与自动降级。

## 常量

- `HAS_RUST_CORE` (bool) — Rust 扩展是否可用

## 函数

- `rust_core_status()` — 返回 Rust 8 组件可用性字典
- `parallel_lif_tick_safe(voltages, *, v_rest, v_reset, v_th, tau_m, r_m, inputs, dt)` — 并行 LIF 更新，返回 (voltages, spikes, backend)
- `mine_pattern_counts_safe(samples, min_support)` — N-gram 模式挖掘，返回 (trigram_counts, bigram_counts, backend)

当 Rust 扩展不可用时，所有函数自动回退到 Python 实现。

::: neurogenesis_lab.core_bridge
