from __future__ import annotations

import sys
from typing import Any


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _probe_components(status: dict[str, Any]) -> None:
    required = [
        "TickSnapshot",
        "EngineSummary",
        "LIFCore",
        "IzhikevichCore",
        "SparseAdjacency",
        "EventQueue",
        "parallel_lif_tick",
        "mine_pattern_counts",
    ]
    components = status.get("components", {})
    missing = [name for name in required if not bool(components.get(name, False))]
    if missing:
        _fail(f"Rust 组件缺失: {', '.join(missing)}")
    _ok("Rust 组件探测通过")


def _probe_runtime_backend() -> None:
    from neurogenesis_lab.core_bridge import (
        mine_pattern_counts_safe,
        parallel_lif_tick_safe,
    )

    _, _, lif_backend = parallel_lif_tick_safe(
        [-65.0, -64.5, -63.2],
        v_rest=-65.0,
        v_reset=-65.0,
        v_th=-55.0,
        tau_m=20.0,
        r_m=1.0,
        inputs=[4.0, 3.0, 2.0],
        dt=1.0,
    )
    if lif_backend != "rust":
        _fail(f"parallel_lif_tick backend 异常: {lif_backend}")

    _, _, mine_backend = mine_pattern_counts_safe(
        [[1, 2, 3, 4], [1, 2, 3], [2, 3, 4]],
        min_support=1,
    )
    if mine_backend != "rust":
        _fail(f"mine_pattern_counts backend 异常: {mine_backend}")

    _ok("运行时 backend 探测通过（非 python-fallback）")


def main() -> int:
    try:
        from neurogenesis_lab.core_bridge import rust_core_status
    except Exception as exc:  # pragma: no cover
        _fail(f"导入 core_bridge 失败: {exc}")

    status = rust_core_status()
    if not bool(status.get("available", False)):
        _fail("Rust Core 不可用（available=False）")

    _ok("Rust Core: available")
    _probe_components(status)
    _probe_runtime_backend()
    print("[PASS] 发布前 Rust 运行时检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
