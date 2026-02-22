"""单元测试 — Rust 核心桥接层"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from neurogenesis_lab import core_bridge


def test_rust_core_status_has_expected_keys() -> None:
    status = core_bridge.rust_core_status()
    assert "schema_version" in status
    assert "available" in status
    assert "components" in status
    assert "parallel_lif_tick" in status["components"]
    assert "mine_pattern_counts" in status["components"]


def test_parallel_lif_tick_safe_python_fallback(monkeypatch) -> None:
    monkeypatch.setattr(core_bridge, "HAS_RUST_CORE", False)
    out_v, spikes, backend = core_bridge.parallel_lif_tick_safe(
        [-65.0, -60.0],
        v_rest=-65.0,
        v_reset=-65.0,
        v_th=-55.0,
        tau_m=20.0,
        r_m=1.0,
        inputs=[0.0, 100.0],
        dt=1.0,
    )
    assert backend == "python-fallback"
    assert len(out_v) == 2
    assert len(spikes) == 2


def test_mine_pattern_counts_safe_python_fallback(monkeypatch) -> None:
    monkeypatch.setattr(core_bridge, "HAS_RUST_CORE", False)
    trigram, bigram, backend = core_bridge.mine_pattern_counts_safe(
        [[1, 2, 3], [1, 2, 3], [1, 2]],
        min_support=2,
    )
    assert backend == "python-fallback"
    assert trigram.get((1, 2, 3)) == 2
    assert bigram.get((1, 2)) == 3
