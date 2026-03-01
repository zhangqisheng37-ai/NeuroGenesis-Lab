"""单元测试 — vectorized 引擎"""


import numpy as np
import pytest
from neurogenesis_lab.encoding import Event
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
        connections = net.weight_matrix.nnz
        assert connections > 0, "kNN 连接应产生连接"

    def test_connect_radius(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(20, spread=5.0)
        net.connect_radius(radius=8.0, weight=0.5)
        connections = net.weight_matrix.nnz
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
        assert "total_propagations" in summary
        assert "active_ratio" in summary
        assert summary["total_propagations"] >= 0
        assert 0.0 <= summary["active_ratio"] <= 1.0

    def test_tick_snapshot_has_propagation_count(self) -> None:
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(5, v_th=-64.0, r_m=30.0, tau_m=1.0)
        net.connect_knn(k=2)
        currents = np.full(5, 10.0)
        snapshot = net.tick(1, input_currents=currents)
        assert "propagation_count" in snapshot
        assert snapshot["propagation_count"] >= 0

    def test_delayed_routing_respects_delay(self) -> None:
        """放电传播应按 delay 在后续 tick 到达。"""
        np.random.seed(42)
        net = VectorizedNetwork(
            energy_decay=0.0,
            move_cost=0.0,
            input_current_scale=5.0,
            reproduction_threshold=999.0,
            min_neurons=0,
        )
        net.init_neurons(2, v_th=-64.0, r_m=30.0, tau_m=1.0, energy_init=2.0)
        net.connect_knn(k=1, weight=1.0, delay=2)

        s1 = net.tick(1, input_currents=np.array([1.0, 0.0], dtype=np.float64))
        assert s1["tick_spikes"] >= 1
        assert s1["propagation_count"] >= 1

        s2 = net.tick(2)
        assert s2.get("ready_events", 0) == 0

        s3 = net.tick(3)
        assert s3.get("ready_events", 0) >= 1

    def test_enqueue_external_event_is_consumed(self) -> None:
        """外部 enqueue 事件应在指定 tick 注入目标神经元。"""
        np.random.seed(42)
        net = VectorizedNetwork(
            energy_decay=0.0,
            move_cost=0.0,
            input_current_scale=5.0,
            reproduction_threshold=999.0,
            min_neurons=0,
        )
        net.init_neurons(3, v_th=-64.0, r_m=30.0, tau_m=1.0, energy_init=2.0)

        net.enqueue(
            [
                Event(token_id=1, t=2, src_id="1", meta={"weight": 1.0}),
                Event(token_id=1, t=2, src_id="bad-id", meta={"weight": 1.0}),
            ]
        )

        s1 = net.tick(1)
        assert s1.get("ready_events", 0) == 0

        s2 = net.tick(2)
        assert s2.get("ready_events", 0) >= 1
        assert s2["tick_spikes"] >= 1

    def test_nan_protection(self) -> None:
        """NaN 输入不应导致崩溃"""
        np.random.seed(42)
        net = VectorizedNetwork()
        net.init_neurons(5)
        currents = np.array([float("nan"), 0.0, float("inf"), 0.0, -float("inf")])
        # np.nan 进入后 safe 处理
        net.tick(1, input_currents=currents)
        assert np.all(np.isfinite(net.v[net.alive])), "存活神经元不应有 NaN/Inf 电位"

    def test_tick_snapshot_sequence_regression(self) -> None:
        np.random.seed(123)
        net = VectorizedNetwork(
            energy_decay=0.0,
            move_cost=0.0,
            input_current_scale=5.0,
            reproduction_threshold=999.0,
            min_neurons=0,
        )
        net.init_neurons(2, v_th=-64.0, r_m=30.0, tau_m=1.0, energy_init=2.0)
        net.connect_knn(k=1, weight=1.0, delay=2)

        snapshots = [
            net.tick(1, input_currents=np.array([1.0, 0.0], dtype=np.float64)),
            net.tick(2),
            net.tick(3),
        ]
        expected = [
            {
                "tick": 1,
                "ready_events": 0,
                "propagation_count": 1,
                "alive_neurons": 2,
                "tick_spikes": 1,
                "births": 0,
                "deaths": 0,
                "prediction_error": 0.0,
                "mean_energy": 1.975,
            },
            {
                "tick": 2,
                "ready_events": 0,
                "propagation_count": 0,
                "alive_neurons": 2,
                "tick_spikes": 0,
                "births": 0,
                "deaths": 0,
                "prediction_error": 0.0,
                "mean_energy": 1.975,
            },
            {
                "tick": 3,
                "ready_events": 1,
                "propagation_count": 1,
                "alive_neurons": 2,
                "tick_spikes": 1,
                "births": 0,
                "deaths": 0,
                "prediction_error": 0.025000000000000133,
                "mean_energy": 1.95,
            },
        ]
        for snapshot, expected_tick in zip(snapshots, expected):
            for key, value in expected_tick.items():
                if isinstance(value, float):
                    assert snapshot[key] == pytest.approx(value)
                else:
                    assert snapshot[key] == value

    def test_scheduled_inputs_cleanup_processed_ticks(self) -> None:
        net = VectorizedNetwork(
            energy_decay=0.0,
            move_cost=0.0,
            input_current_scale=0.0,
            reproduction_threshold=999.0,
            min_neurons=0,
        )
        net.init_neurons(2, energy_init=2.0)
        net.scheduled_inputs[0] = np.array([0.5, 0.0], dtype=np.float64)
        net.scheduled_inputs[2] = np.array([0.0, 0.5], dtype=np.float64)

        snapshot = net.tick(2)

        assert snapshot["ready_events"] >= 1
        assert 0 not in net.scheduled_inputs
        assert 2 not in net.scheduled_inputs
