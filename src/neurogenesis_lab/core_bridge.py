"""Rust 核心桥接层：统一导入并提供降级能力。"""

from __future__ import annotations

import types
from typing import Any
import importlib

from neurogenesis_lab.config import SCHEMA_VERSION

core: types.ModuleType | None
try:
    core = importlib.import_module("neurogenesis_lab._neurogenesis_core")

    HAS_RUST_CORE = True
except Exception:  # pragma: no cover - 依赖本地构建环境
    core = None
    HAS_RUST_CORE = False


def _core_has(symbol: str) -> bool:
    return bool(HAS_RUST_CORE and hasattr(core, symbol))


def rust_core_status() -> dict[str, Any]:
    components = {
        "TickSnapshot": _core_has("TickSnapshot"),
        "EngineSummary": _core_has("EngineSummary"),
        "LIFCore": _core_has("LIFCore"),
        "IzhikevichCore": _core_has("IzhikevichCore"),
        "SparseAdjacency": _core_has("SparseAdjacency"),
        "EventQueue": _core_has("EventQueue"),
        "parallel_lif_tick": _core_has("parallel_lif_tick"),
        "mine_pattern_counts": _core_has("mine_pattern_counts"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "available": HAS_RUST_CORE,
        "module": "neurogenesis_lab._neurogenesis_core" if HAS_RUST_CORE else None,
        "components": components,
    }


def parallel_lif_tick_safe(
    voltages: list[float],
    *,
    v_rest: float,
    v_reset: float,
    v_th: float,
    tau_m: float,
    r_m: float,
    inputs: list[float],
    dt: float,
) -> tuple[list[float], list[bool], str]:
    """并行 LIF 更新包装。

    返回: (更新后电位, 放电脉冲掩码, backend)
    backend 为 "rust" 或 "python-fallback"。
    """
    if len(voltages) != len(inputs):
        raise ValueError("voltages and inputs length mismatch")
    if dt <= 0.0 or tau_m <= 0.0:
        raise ValueError("dt and tau_m must be positive")

    core_mod = core
    if core_mod is not None and HAS_RUST_CORE and hasattr(core_mod, "parallel_lif_tick"):
        out_v, spikes = core_mod.parallel_lif_tick(
            voltages,
            v_rest,
            v_reset,
            v_th,
            tau_m,
            r_m,
            inputs,
            dt,
        )
        return list(out_v), list(spikes), "rust"

    out_v = list(voltages)
    spikes = [False] * len(out_v)
    for idx in range(len(out_v)):
        dv = (-(out_v[idx] - v_rest) + r_m * inputs[idx]) / tau_m * dt
        out_v[idx] += dv
        if out_v[idx] >= v_th:
            out_v[idx] = v_reset
            spikes[idx] = True
    return out_v, spikes, "python-fallback"


def mine_pattern_counts_safe(
    samples: list[list[int]],
    min_support: int,
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int], int], str]:
    """N-gram 模式挖掘包装。

    返回: (trigram_support, bigram_support, backend)
    backend 为 "rust" 或 "python-fallback"。
    """
    if min_support < 1:
        raise ValueError("min_support must be >= 1")

    core_mod = core
    if core_mod is not None and HAS_RUST_CORE and hasattr(core_mod, "mine_pattern_counts"):
        tri_rows, bi_rows = core_mod.mine_pattern_counts(samples, min_support)
        rust_trigram_support = {
            (int(a), int(b), int(c)): int(support) for a, b, c, support in tri_rows
        }
        rust_bigram_support = {
            (int(a), int(b)): int(support) for a, b, support in bi_rows
        }
        return rust_trigram_support, rust_bigram_support, "rust"

    trigram_support: dict[tuple[int, int, int], int] = {}
    bigram_support: dict[tuple[int, int], int] = {}
    for sequence in samples:
        trigrams = {
            (sequence[index], sequence[index + 1], sequence[index + 2])
            for index in range(max(0, len(sequence) - 2))
        }
        bigrams = {
            (sequence[index], sequence[index + 1])
            for index in range(max(0, len(sequence) - 1))
        }
        for trigram in trigrams:
            trigram_support[trigram] = trigram_support.get(trigram, 0) + 1
        for bigram in bigrams:
            bigram_support[bigram] = bigram_support.get(bigram, 0) + 1

    trigram_support = {
        trigram: support
        for trigram, support in trigram_support.items()
        if support >= min_support
    }
    bigram_support = {
        bigram: support
        for bigram, support in bigram_support.items()
        if support >= min_support
    }
    return trigram_support, bigram_support, "python-fallback"
