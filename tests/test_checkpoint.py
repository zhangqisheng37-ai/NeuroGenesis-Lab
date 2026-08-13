"""单元测试 — checkpoint 持久化与恢复"""

import pickle
import random
from pathlib import Path

import pytest

from neurogenesis_lab.config import SCHEMA_VERSION
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


def test_checkpoint_truncated_pickle_raises_diagnostic(tmp_path: Path) -> None:
    """截断的 pickle（磁盘写一半断电/空间不足）应抛出可诊断异常而非静默。"""
    net = _build_seeded_network(seed=3)
    checkpoint = tmp_path / "truncated.pkl"
    net.save_checkpoint(checkpoint)

    data = checkpoint.read_bytes()
    checkpoint.write_bytes(data[: len(data) // 2])

    with pytest.raises((EOFError, pickle.UnpicklingError)):
        Network.load_checkpoint(checkpoint)


def test_checkpoint_non_dict_payload_rejected(tmp_path: Path) -> None:
    """顶层非 dict（如 list/str）应被拒绝并给出可诊断信息。"""
    checkpoint = tmp_path / "not_dict.pkl"
    with checkpoint.open("wb") as f:
        pickle.dump(["not", "a", "dict"], f)

    with pytest.raises(ValueError, match="顶层必须为 dict"):
        Network.load_checkpoint(checkpoint)


def test_checkpoint_missing_network_object_rejected(tmp_path: Path) -> None:
    """dict 但缺少 network 对象（损坏/手工构造）应被拒绝。"""
    checkpoint = tmp_path / "no_network.pkl"
    with checkpoint.open("wb") as f:
        pickle.dump({"schema_version": SCHEMA_VERSION, "tick": 5}, f)

    with pytest.raises(ValueError, match="缺少 network 对象"):
        Network.load_checkpoint(checkpoint)


def test_checkpoint_missing_optional_keys_restores(tmp_path: Path) -> None:
    """缺 tick/random_state 等可选键时按防御默认值恢复，不崩溃。

    缺 tick 键时回退到网络自身的 current_tick；缺 random_state 时不恢复随机状态。
    """
    net = _build_seeded_network(seed=5)
    net.tick(3)
    checkpoint = tmp_path / "minimal.pkl"
    payload = {"schema_version": SCHEMA_VERSION, "network": net}
    with checkpoint.open("wb") as f:
        pickle.dump(payload, f)

    restored = Network.load_checkpoint(checkpoint)
    assert restored.current_tick == 3, "缺 tick 键时应回退到网络自身 current_tick"
    assert len(restored.neurons) == len(net.neurons)




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
