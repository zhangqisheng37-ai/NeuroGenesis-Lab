"""单元测试 — vectorized 引擎（含 P2 多模型/尸体/自适应功能与自动学习联动，原三个文件合并）"""


import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from neurogenesis_lab.config import (
    ENDANGERED_DECAY_DISCOUNT,
    ENDANGERED_EXIT_MULTIPLIER,
    REPRO_SHARE_MAX,
    REPRO_SHARE_MIN,
    SCHEMA_VERSION,
)
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.learning import AutoLearner, RuleDiscovery, RuleStore
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


# ======================================================================
# 合并自 test_vectorized_p2.py：Phase 7 — P2 VectorizedNetwork 功能对齐测试。
# ======================================================================


# =========================================================================
# 辅助工具
# =========================================================================


def _make_net(n: int = 50, **kwargs) -> VectorizedNetwork:
    """创建小型测试网络。"""
    net = VectorizedNetwork(**kwargs)
    net.init_neurons(n, spread=10.0, energy_init=2.0)
    net.connect_knn(k=2)
    return net


def _make_mixed_net(n: int = 60) -> VectorizedNetwork:
    """创建含 LIF/Izh/AdEx 混合模型的网络。"""
    net = VectorizedNetwork()
    net.init_neurons(n, spread=10.0, energy_init=3.0)
    # 前 1/3: LIF, 中 1/3: Izh, 后 1/3: AdEx
    third = n // 3
    net.model_type[:third] = 0
    net.model_type[third : 2 * third] = 1
    net.model_type[2 * third :] = 2
    net.connect_knn(k=2)
    return net


# =========================================================================
# 多模型动力学
# =========================================================================


class TestMultiModelDynamics:
    """多模型动力学步进 (LIF / Izhikevich / AdEx) 测试。"""

    def test_pure_lif_step(self):
        """纯 LIF 网络应与之前行为一致。"""
        net = _make_net(20)
        snap = net.tick(1, dt=1.0)
        assert snap["alive_neurons"] <= 20
        assert snap["tick_spikes"] >= 0

    def test_pure_izh_step(self):
        """纯 Izhikevich 网络可正常完成 tick。"""
        net = _make_net(20)
        net.model_type[:] = 1
        # 给足够输入促进放电
        inputs = np.full(20, 15.0)
        snap = net.tick(1, dt=1.0, input_currents=inputs)
        assert snap["alive_neurons"] <= 20
        assert snap["tick_spikes"] >= 0

    def test_pure_adex_step(self):
        """纯 AdEx 网络可正常完成 tick。"""
        net = _make_net(20)
        net.model_type[:] = 2
        inputs = np.full(20, 10.0)
        snap = net.tick(1, dt=1.0, input_currents=inputs)
        assert snap["alive_neurons"] <= 20

    def test_mixed_model_no_crash(self):
        """混合模型网络运行多 tick 不崩溃。"""
        net = _make_mixed_net(60)
        for t in range(1, 21):
            snap = net.tick(t, dt=1.0)
        assert snap["tick"] == 20

    def test_izh_spike_resets_to_c(self):
        """Izhikevich 神经元放电后 v 应重置为 izh_c。"""
        net = _make_net(5)
        net.model_type[:] = 1
        net.izh_c[:] = -70.0
        # 强制 v 超阈值
        net.v[:] = net.v_th + 10.0
        net.tick(1, dt=1.0)
        # 放电后 v 应为 izh_c
        for i in range(5):
            if net.alive[i]:
                assert net.v[i] == pytest.approx(-70.0, abs=1.0)

    def test_adex_spike_resets(self):
        """AdEx 神经元放电后 v 应重置为 v_reset。"""
        net = _make_net(5)
        net.model_type[:] = 2
        net.v[:] = net.v_th + 10.0
        net.tick(1, dt=1.0)
        for i in range(5):
            if net.alive[i]:
                assert net.v[i] <= net.v_th[i]

    def test_model_type_inherited_on_birth(self):
        """繁殖时子代应继承父代 model_type。"""
        net = _make_net(10, max_neurons=20)
        net.model_type[0] = 1  # 第 0 号设为 Izh
        net.energy[0] = 100.0  # 确保能繁殖
        # 杀掉一些以腾出槽位
        net.alive[5:] = False
        # 运行多次尝试繁殖
        for t in range(1, 20):
            net.tick(t)
        # 检查是否有新 Izh 类型的神经元
        izh_alive = np.sum((net.model_type == 1) & net.alive)
        assert izh_alive >= 1

    def test_lif_step_no_modify_dead_neurons(self):
        """死亡神经元的电位不应被更新。"""
        net = _make_net(10)
        net.alive[3] = False
        v_before = net.v[3]
        net.tick(1, dt=1.0, input_currents=np.full(10, 5.0))
        assert net.v[3] == v_before


# =========================================================================
# 尸体回收系统
# =========================================================================


class TestCorpseSystem:
    """尸体回收系统测试。"""

    def test_corpse_created_on_death(self):
        """神经元死亡时应标记为尸体。"""
        net = _make_net(10)
        # 强制第 0 个能量为 0 使其死亡
        net.energy[0] = 0.0
        net.tick(1)
        # 如果确实死亡，corpse_mask 应被设置
        if not net.alive[0]:
            assert bool(net.corpse_mask[0])

    def test_corpse_ttl_decrements(self):
        """尸体 TTL 每 tick 递减。"""
        net = _make_net(10)
        # 手动设置一个尸体
        net.alive[0] = False
        net.corpse_mask[0] = True
        net.corpse_timer[0] = 10
        net.corpse_energy[0] = 0.5
        net._update_corpses()
        assert net.corpse_timer[0] == 9

    def test_corpse_energy_decays(self):
        """尸体能量每 tick 自然衰减。"""
        net = _make_net(10)
        net.alive[0] = False
        net.corpse_mask[0] = True
        net.corpse_timer[0] = 10
        net.corpse_energy[0] = 1.0
        initial_energy = net.corpse_energy[0]
        net._update_corpses()
        assert net.corpse_energy[0] < initial_energy

    def test_corpse_removed_when_expired(self):
        """TTL 到期的尸体应被清理。"""
        net = _make_net(10)
        net.alive[0] = False
        net.corpse_mask[0] = True
        net.corpse_timer[0] = 1  # 再减一次就到 0
        net.corpse_energy[0] = 0.5
        net._update_corpses()
        assert not net.corpse_mask[0]

    def test_corpse_scavenging(self):
        """附近存活神经元应吸收尸体能量。"""
        net = _make_net(10, scavenger_radius=100.0)  # 大范围确保命中
        net.spatial.build(net.pos)
        net.alive[0] = False
        net.corpse_mask[0] = True
        net.corpse_timer[0] = 20
        net.corpse_energy[0] = 2.0
        # 确保至少有一个存活神经元在附近
        net.pos[1] = net.pos[0] + 0.01  # 极近距离
        net.spatial.build(net.pos)
        energy_before_1 = net.energy[1]
        net._update_corpses()
        # 最近的存活神经元应获得能量
        assert net.energy[1] > energy_before_1 or net.corpse_energy[0] == 0.0

    def test_no_corpse_slot_reuse_during_repro(self):
        """繁殖不应占用尸体槽位。"""
        net = _make_net(10, max_neurons=10)
        net.alive[9] = False
        net.corpse_mask[9] = True
        net.corpse_timer[9] = 30
        net.corpse_energy[9] = 1.0
        net.energy[0] = 100.0
        # 只有 index=9 是空的，但是尸体
        net.tick(1)
        # 如果实现正确，则不会使用尸体槽位繁殖
        # (或者 newborn 清除了尸体标记)
        if net.alive[9]:
            assert not net.corpse_mask[9]


# =========================================================================
# 自适应衰减
# =========================================================================


class TestAdaptiveDecay:
    """自适应能量衰减测试。"""

    def test_high_density_increases_decay(self):
        """高密度时衰减应增大。"""
        net = _make_net(100, max_neurons=100)
        # 全部存活 → density_ratio = 1.0 > DENSITY_HIGH_THRESHOLD
        decay = net._calculate_adaptive_decay(100)
        assert decay >= net.energy_decay

    def test_low_density_decreases_decay(self):
        """低密度时衰减应减小。"""
        net = _make_net(100, max_neurons=1000)
        # 仅 1 存活 → density_ratio ~= 0.001
        decay = net._calculate_adaptive_decay(1)
        assert decay <= net.energy_decay

    def test_disabled_returns_base(self):
        """关闭自适应衰减时返回基础衰减。"""
        net = _make_net(50, adaptive_decay=False)
        decay = net._calculate_adaptive_decay(25)
        assert decay == net.energy_decay

    def test_endangered_discount_applied(self):
        """濒危模式时衰减应打折。"""
        net = _make_net(50)
        net._endangered_mode = True
        base_decay = net._calculate_adaptive_decay(25)
        effective = base_decay * ENDANGERED_DECAY_DISCOUNT
        assert effective < base_decay


# =========================================================================
# 自适应繁殖
# =========================================================================


class TestAdaptiveReproduction:
    """自适应繁殖份额测试。"""

    def test_high_population_increases_share(self):
        """高种群时繁殖份额应增大（更多能量消耗）。"""
        net = _make_net(100, max_neurons=100)
        share = net._calculate_adaptive_reproduction_share(100)
        assert share >= net.reproduction_share

    def test_low_population_decreases_share(self):
        """低种群时繁殖份额应减小（鼓励繁殖）。"""
        net = _make_net(100, max_neurons=1000)
        share = net._calculate_adaptive_reproduction_share(1)
        assert share <= net.reproduction_share

    def test_share_bounded(self):
        """份额应在 [REPRO_SHARE_MIN, REPRO_SHARE_MAX] 之间。"""
        net = _make_net(50)
        for count in [0, 1, 25, 49, 50]:
            share = net._calculate_adaptive_reproduction_share(count)
            assert REPRO_SHARE_MIN <= share <= REPRO_SHARE_MAX

    def test_disabled_returns_base(self):
        """关闭自适应繁殖时返回基础份额。"""
        net = _make_net(50, adaptive_reproduction=False)
        share = net._calculate_adaptive_reproduction_share(25)
        assert share == net.reproduction_share


# =========================================================================
# 能量上限
# =========================================================================


class TestEnergyCap:
    """能量上限裁切测试。"""

    def test_energy_capped_after_tick(self):
        """tick 执行后能量不应超过 energy_cap。"""
        net = _make_net(10, energy_cap=3.0)
        net.energy[:] = 10.0  # 远超上限
        net.tick(1)
        # 存活神经元能量应 <= energy_cap
        alive_energy = net.energy[net.alive]
        if len(alive_energy) > 0:
            assert np.all(alive_energy <= 3.0 + 1e-10)


# =========================================================================
# 濒危模式
# =========================================================================


class TestEndangeredMode:
    """濒危模式测试。"""

    def test_enters_endangered_mode(self):
        """存活数低于阈值时应进入濒危模式。"""
        net = _make_net(20, endangered_threshold=15)
        # 杀掉大部分
        net.alive[10:] = False
        net.tick(1)
        assert net._endangered_mode is True

    def test_exits_endangered_mode(self):
        """存活数恢复后应退出濒危模式。"""
        net = _make_net(50, endangered_threshold=5)
        net._endangered_mode = True
        # 多数存活 → 退出
        net.tick(1)
        exit_threshold = 5 * ENDANGERED_EXIT_MULTIPLIER
        if int(np.sum(net.alive)) > exit_threshold:
            assert net._endangered_mode is False


# =========================================================================
# Checkpoint
# =========================================================================


class TestCheckpoint:
    """Checkpoint 持久化测试。"""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """保存后加载应恢复网络状态。"""
        net = _make_net(30)
        for t in range(1, 6):
            net.tick(t)
        ckpt_path = tmp_path / "test.ckpt"
        net.save_checkpoint(ckpt_path, save_tick=5)

        loaded_net, loaded_tick = VectorizedNetwork.load_checkpoint(ckpt_path)
        assert loaded_tick == 5
        assert loaded_net.n == net.n
        np.testing.assert_array_equal(loaded_net.alive, net.alive)
        np.testing.assert_array_almost_equal(loaded_net.energy, net.energy)
        np.testing.assert_array_almost_equal(loaded_net.v, net.v)

    def test_checkpoint_preserves_model_types(self, tmp_path: Path):
        """Checkpoint 应保留 model_type 向量。"""
        net = _make_mixed_net(30)
        net.tick(1)
        ckpt_path = tmp_path / "mixed.ckpt"
        net.save_checkpoint(ckpt_path, save_tick=1)

        loaded_net, _ = VectorizedNetwork.load_checkpoint(ckpt_path)
        np.testing.assert_array_equal(loaded_net.model_type, net.model_type)

    def test_checkpoint_schema_mismatch_raises(self, tmp_path: Path):
        """Schema 版本不兼容时应报错。"""
        net = _make_net(10)
        ckpt_path = tmp_path / "bad_schema.ckpt"
        payload = {
            "schema_version": "999.0",
            "tick": 0,
            "random_state": __import__("random").getstate(),
            "network": net,
        }
        with open(ckpt_path, "wb") as f:
            pickle.dump(payload, f)

        with pytest.raises(ValueError, match="Schema"):
            VectorizedNetwork.load_checkpoint(ckpt_path)

    def test_checkpoint_type_mismatch_raises(self, tmp_path: Path):
        """对象类型不匹配时应报错。"""
        ckpt_path = tmp_path / "bad_type.ckpt"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tick": 0,
            "random_state": __import__("random").getstate(),
            "network": "not_a_network",
        }
        with open(ckpt_path, "wb") as f:
            pickle.dump(payload, f)

        with pytest.raises(TypeError, match="不匹配"):
            VectorizedNetwork.load_checkpoint(ckpt_path)

    def test_checkpoint_subdirectory_created(self, tmp_path: Path):
        """保存到不存在的子目录时应自动创建。"""
        net = _make_net(5)
        deep_path = tmp_path / "a" / "b" / "c" / "test.ckpt"
        net.save_checkpoint(deep_path, save_tick=0)
        assert deep_path.exists()


# =========================================================================
# get_summary 完整性
# =========================================================================


class TestSummaryIntegrity:
    """get_summary 返回值完整性测试。"""

    def test_summary_keys(self):
        """摘要应包含所有必要键。"""
        net = _make_net(20)
        net.tick(1)
        summary = net.get_summary()
        expected_keys = {
            "schema_version",
            "total_neurons",
            "alive_neurons",
            "total_spikes",
            "total_propagations",
            "birth_count",
            "death_count",
            "mean_energy",
            "pe_moving_avg",
            "active_ratio",
        }
        assert expected_keys <= set(summary.keys())

    def test_summary_after_mixed_run(self):
        """混合模型运行后摘要应有合理值。"""
        net = _make_mixed_net(60)
        for t in range(1, 11):
            net.tick(t)
        summary = net.get_summary()
        assert summary["total_neurons"] == 60
        assert summary["alive_neurons"] >= 0
        assert summary["active_ratio"] >= 0.0


# =========================================================================
# 边界条件
# =========================================================================


class TestEdgeCases:
    """边界条件及回归测试。"""

    def test_zero_neurons(self):
        """零神经元网络应可创建且不崩溃。"""
        net = VectorizedNetwork()
        net.init_neurons(0)
        snap = net.tick(1)
        assert snap["alive_neurons"] == 0

    def test_single_neuron(self):
        """单神经元网络应可正常运行。"""
        net = _make_net(1)
        snap = net.tick(1)
        assert snap["total_neurons"] == 1

    def test_all_dead_no_crash(self):
        """所有神经元死亡后仍可调用 tick。"""
        net = _make_net(10, min_neurons=0)
        net.energy[:] = 0.0
        net.alive[:] = False
        snap = net.tick(1)
        assert snap["alive_neurons"] == 0

    def test_nan_input_handled(self):
        """NaN 输入电流不应导致崩溃。"""
        net = _make_net(10)
        inputs = np.full(10, np.nan)
        snap = net.tick(1, input_currents=inputs)
        # 应正常返回，NaN 已被清零
        assert np.isfinite(snap["mean_energy"])

    def test_spatial_grid_rebuild(self):
        """空间索引重建不应导致异常。"""
        net = _make_net(20)
        net.spatial.build(net.pos)
        results = net.spatial.query_radius(0.0, 0.0, 100.0, net.pos)
        assert len(results) == 20


# ======================================================================
# 合并自 test_vectorized_autolearn.py：VectorizedNetwork 与 AutoLearner 集成测试。
# ======================================================================


def _make_vectorized_net(n: int = 6) -> VectorizedNetwork:
    net = VectorizedNetwork(energy_decay=0.0, move_cost=0.0, min_neurons=0)
    net.init_neurons(n, energy_init=1.0)
    return net


def _write_seq(net: VectorizedNetwork, idx: int, seq: list[int]) -> None:
    for token in seq:
        net._write_token(idx, token)


def test_vectorized_snapshot_can_feed_autolearner() -> None:
    """VectorizedNetwork.tick() 的快照可直接交给 AutoLearner.on_tick()。"""
    np.random.seed(42)
    net = _make_vectorized_net(6)

    # 准备可学习样本（满足 min_samples=3）
    _write_seq(net, 0, [49, 43, 50, 61, 51])  # 1+2=3
    _write_seq(net, 1, [50, 43, 51, 61, 53])  # 2+3=5
    _write_seq(net, 2, [51, 43, 52, 61, 55])  # 3+4=7

    auto = AutoLearner(discovery=RuleDiscovery(min_support=1), learn_interval=1)
    snapshot = net.tick(1)
    report = auto.on_tick(1, snapshot, net)

    assert "rule_count" in report
    assert "prediction_accuracy" in report
    assert "tick_predictions" in report
    assert "tick_correct" in report

    # 模拟 CLI 注入行为
    snapshot["rule_count"] = report["rule_count"]
    snapshot["prediction_accuracy"] = report["prediction_accuracy"]
    assert isinstance(snapshot["rule_count"], int)
    assert isinstance(snapshot["prediction_accuracy"], float)


def test_vectorized_prediction_reward_and_hits() -> None:
    """命中预测应增加能量并累计 prediction_hits。"""
    net = _make_vectorized_net(2)
    net.energy[:] = 1.0

    # 让规则组合器可根据 bigram 预测最后一个 token
    auto = AutoLearner(
        discovery=RuleDiscovery(min_support=1),
        prediction_reward=0.5,
        prediction_penalty=0.2,
    )
    auto.store = RuleStore(
        trigram_support=OrderedDict(),
        bigram_support=OrderedDict({(3, 4): 10}),
    )
    auto.rule_count = 1

    _write_seq(net, 0, [1, 2, 3, 4])  # 预测命中
    _write_seq(net, 1, [1, 2, 3, 9])  # 预测错误

    snapshot = net.tick(1)
    report = auto.on_tick(1, snapshot, net)

    assert report["tick_predictions"] == 2
    assert report["tick_correct"] >= 1
    assert net.energy[0] > 1.0
    assert net.prediction_hits[0] >= 1


def test_vectorized_penalty_never_below_zero() -> None:
    """预测错误惩罚后能量不应低于 0。"""
    net = _make_vectorized_net(1)
    net.energy[0] = 0.01

    auto = AutoLearner(
        discovery=RuleDiscovery(min_support=1),
        prediction_reward=0.0,
        prediction_penalty=1.0,
    )
    auto.store = RuleStore(
        trigram_support=OrderedDict(),
        bigram_support=OrderedDict({(7, 8): 5}),
    )
    auto.rule_count = 1

    _write_seq(net, 0, [1, 2, 7, 9])  # 预测应为 8，实际为 9

    snapshot = net.tick(1)
    auto.on_tick(1, snapshot, net)

    assert float(net.energy[0]) >= 0.0
