"""单元测试 — checkpoint 持久化与恢复"""

import pickle
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

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


def test_resume_matches_continuous_run(tmp_path: Path) -> None:
    base = _build_seeded_network(seed=123)

    # 从同一初始状态克隆两条分支
    baseline: Network = pickle.loads(pickle.dumps(base))
    resumed_branch: Network = pickle.loads(pickle.dumps(base))

    # 连续运行 200 tick
    for t in range(1, 201):
        baseline.tick(t)

    # 运行 100 tick -> 保存 -> 恢复 -> 再运行 100 tick
    for t in range(1, 101):
        resumed_branch.tick(t)
    checkpoint = tmp_path / "resume.pkl"
    resumed_branch.save_checkpoint(checkpoint, tick=100)

    restored = Network.load_checkpoint(checkpoint)
    for t in range(101, 201):
        restored.tick(t)

    baseline_summary = baseline.get_summary()
    restored_summary = restored.get_summary()

    assert restored.current_tick == 200
    assert restored_summary["total_neurons"] == baseline_summary["total_neurons"]
    assert restored_summary["alive_neurons"] == baseline_summary["alive_neurons"]
    assert restored_summary["total_spikes"] == baseline_summary["total_spikes"]
    assert (
        restored_summary["total_propagations"] == baseline_summary["total_propagations"]
    )
    assert restored_summary["birth_count"] == baseline_summary["birth_count"]
    assert restored_summary["death_count"] == baseline_summary["death_count"]
