"""单元测试 — checkpoint 持久化与恢复"""

import pickle
import random
from pathlib import Path

import pytest

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron


def _build_seeded_network(seed: int = 42) -> Network:
    random.seed(seed)
    n1 = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
    n2 = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
    net = Network(
        neurons={n1.id: n1, n2.id: n2},
        input_current_scale=5.0,
        max_neurons=20,
        min_neurons=0,
    )
    net.connect(n1.id, n2.id, weight=1.0, delay=1)
    for t in range(1, 401):
        net.enqueue([Event(token_id=1, t=t, src_id=n1.id, meta={"weight": 1.0})])
    return net


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    net = _build_seeded_network(seed=7)
    for t in range(1, 11):
        net.tick(t)

    checkpoint = tmp_path / "run_ckpt.pkl"
    saved_path = net.save_checkpoint(checkpoint)
    assert saved_path.exists()

    restored = Network.load_checkpoint(saved_path)
    assert restored.current_tick == 10
    assert len(restored.neurons) == len(net.neurons)
    assert len(restored.queue) == len(net.queue)
    assert restored.birth_count == net.birth_count
    assert restored.death_count == net.death_count


def test_checkpoint_schema_major_mismatch_rejected(tmp_path: Path) -> None:
    net = _build_seeded_network(seed=9)
    net.tick(1)
    checkpoint = tmp_path / "bad_schema.pkl"
    net.save_checkpoint(checkpoint)

    with checkpoint.open("rb") as f:
        payload = pickle.load(f)
    payload["schema_version"] = "999.0.0"
    with checkpoint.open("wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError, match="schema 主版本不兼容"):
        Network.load_checkpoint(checkpoint)




def test_run_checkpoint_resume_summary_consistency(tmp_path: Path) -> None:
    """端到端链路：run -> checkpoint -> resume 与连续运行摘要一致。"""
    seed = 2026

    continuous = _build_seeded_network(seed=seed)
    for t in range(1, 301):
        continuous.tick(t)
    continuous_summary = continuous.get_summary()

    segmented = _build_seeded_network(seed=seed)
    for t in range(1, 151):
        segmented.tick(t)

    checkpoint = tmp_path / "e2e_resume.pkl"
    segmented.save_checkpoint(checkpoint, tick=150)

    restored = Network.load_checkpoint(checkpoint)
    for t in range(151, 301):
        restored.tick(t)
    restored_summary = restored.get_summary()

    assert restored.current_tick == 300
    assert restored_summary["total_neurons"] == continuous_summary["total_neurons"]
    assert restored_summary["alive_neurons"] == continuous_summary["alive_neurons"]
    assert restored_summary["total_spikes"] == continuous_summary["total_spikes"]
    assert restored_summary["total_propagations"] == continuous_summary["total_propagations"]
    assert restored_summary["birth_count"] == continuous_summary["birth_count"]
    assert restored_summary["death_count"] == continuous_summary["death_count"]
