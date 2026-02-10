"""单元测试 — vectorized 引擎"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import numpy as np
import pytest
from neurogenesis_lab.vectorized import SpatialGrid, VectorizedNetwork


class TestSpatialGrid:
    def test_build_and_query(self) -> None:
        grid = SpatialGrid(cell_size=5.0)
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [100.0, 100.0]])
        grid.build(positions)
        result = grid.query_radius(0.0, 0.0, 2.0, positions)
        assert 0 in result
        assert 1 in result
        assert 2 not in result, "远距离点不应在查询结果中"


class TestVectorizedNetwork:
    def test_init_neurons(self) -> None:
        net = VectorizedNetwork()
        net.init_neurons(100)
        assert net.n == 100
        assert len(net.v) == 100
        assert len(net.energy) == 100
        assert np.all(net.alive)

    def test_tick_returns_snapshot(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(10)
        snapshot = net.tick(1)
        assert "tick" in snapshot
        assert "alive_neurons" in snapshot
        assert "schema_version" in snapshot
        assert snapshot["tick"] == 1

    def test_spike_with_current(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(5, v_th=-64.0, r_m=30.0, tau_m=1.0)
        currents = np.full(5, 10.0)
        snapshot = net.tick(1, input_currents=currents)
        assert snapshot["tick_spikes"] > 0, "高电流应触发放电"

    def test_connect_knn(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(10, spread=5.0)
        net.connect_knn(k=3)
        connections = np.sum(net.weight_matrix > 0)
        assert connections > 0, "kNN 连接应产生连接"

    def test_connect_radius(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(20, spread=5.0)
        net.connect_radius(radius=8.0, weight=0.5)
        connections = np.sum(net.weight_matrix > 0)
        assert connections > 0

    def test_energy_decays(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork(energy_decay=0.1, move_cost=0.1)
        net.init_neurons(5, energy_init=1.0)
        initial_total = float(np.sum(net.energy))
        net.tick(1)
        final_total = float(np.sum(net.energy))
        assert final_total < initial_total, "能量应随 tick 衰减"

    def test_death_on_zero_energy(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork(energy_decay=0.5, move_cost=0.5, min_neurons=0)
        net.init_neurons(3, energy_init=0.1)
        net.tick(1)
        alive = int(np.sum(net.alive))
        assert alive < 3, "低能量神经元应死亡"

    def test_min_neurons_protection(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork(energy_decay=10.0, move_cost=10.0, min_neurons=2)
        net.init_neurons(3, energy_init=0.01)
        net.tick(1)
        alive = int(np.sum(net.alive))
        assert alive >= 2, "min_neurons 应保护最少数量"

    def test_reproduction(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork(reproduction_threshold=1.5, max_neurons=20)
        net.init_neurons(10, energy_init=0.5)
        # 前 5 个给高能量，后 5 个标记为死以腾出空位
        net.energy[:5] = 3.0
        net.alive[5:] = False
        net.tick(1)
        assert net.birth_count > 0, "高能量应触发繁殖（死亡空位可用）"

    def test_long_run_stability(self) -> None:
        """长运行测试 — 网络不应崩溃或全灭"""
        np.random.seed(42)
        net = VectorizedNetwork(max_neurons=50, min_neurons=2)
        net.init_neurons(20, energy_init=1.5, spread=10.0)
        net.connect_knn(k=3)
        for t in range(1, 201):
            snapshot = net.tick(t)
        alive = snapshot["alive_neurons"]
        assert alive >= 2, f"200 tick 后应有至少 2 个存活神经元，实际: {alive}"

    def test_get_summary(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(10)
        net.tick(1)
        summary = net.get_summary()
        assert "schema_version" in summary
        assert "total_neurons" in summary
        assert "pe_moving_avg" in summary

    def test_nan_protection(self) -> None:
        """NaN 输入不应导致崩溃"""
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(5)
        currents = np.array([float("nan"), 0.0, float("inf"), 0.0, -float("inf")])
        # np.nan 进入后 safe 处理
        snapshot = net.tick(1, input_currents=currents)
        assert np.all(np.isfinite(net.v[net.alive])), "存活神经元不应有 NaN/Inf 电位"
