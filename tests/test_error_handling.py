"""错误处理测试 — P6.4

覆盖 AsyncJsonlWriter 写入失败/close、Network.check_death min_neurons 边界、
RuleStore LRU 淘汰、visualization matplotlib 异常保护等场景。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import pytest

from neurogenesis_lab.config import RESCUED_NEURON_ENERGY
from neurogenesis_lab.learning import AutoLearner, RuleDiscovery, RuleStore
from neurogenesis_lab.observability import AsyncJsonlWriter


# ══════════════════════════════════════════════════════════
# 1. AsyncJsonlWriter 写入失败测试
# ══════════════════════════════════════════════════════════


class TestAsyncJsonlWriterError:
    """AsyncJsonlWriter 写入失败场景"""

    def test_on_error_callback_called_on_write_failure(self, tmp_path: Path):
        """模拟写入失败时 on_error 回调被调用"""
        error_info: dict = {"called": False, "exc": None, "records": None}

        def on_error(exc: Exception, lost: list[dict]) -> None:
            error_info["called"] = True
            error_info["exc"] = exc
            error_info["records"] = lost

        writer = AsyncJsonlWriter(
            tmp_path / "fail.jsonl",
            buffer_size=1,
            on_error=on_error,
        )

        # 用 mock 让底层 append_many 抛出 IOError
        with patch.object(
            writer._writer, "append_many", side_effect=IOError("磁盘已满")
        ):
            writer.append({"tick": 1})
            # 等待后台线程处理
            time.sleep(1.0)

        writer.close()

        assert error_info["called"], "on_error 回调应被调用"
        assert isinstance(error_info["exc"], IOError)
        assert error_info["records"] is not None
        assert len(error_info["records"]) >= 1

    def test_lost_records_count_correct(self, tmp_path: Path):
        """验证 lost_records 计数正确"""
        writer = AsyncJsonlWriter(
            tmp_path / "fail.jsonl",
            buffer_size=1,
            on_error=lambda exc, lost: None,
        )

        with patch.object(
            writer._writer, "append_many", side_effect=IOError("写入失败")
        ):
            for i in range(5):
                writer.append({"tick": i})
            # 等待后台线程处理所有批次
            time.sleep(2.0)

        writer.close()
        assert writer.lost_records >= 5, (
            f"lost_records 应至少为 5，实际为 {writer.lost_records}"
        )

    def test_no_crash_without_on_error(self, tmp_path: Path, caplog):
        """不提供 on_error 时不崩溃，仅记录日志"""
        writer = AsyncJsonlWriter(
            tmp_path / "fail.jsonl",
            buffer_size=1,
            on_error=None,  # 不提供回调
        )

        with patch.object(
            writer._writer, "append_many", side_effect=IOError("写入失败")
        ):
            writer.append({"tick": 1})
            time.sleep(1.0)

        # 不应崩溃
        writer.close()
        assert writer.lost_records >= 1


# ══════════════════════════════════════════════════════════
# 2. AsyncJsonlWriter close 测试
# ══════════════════════════════════════════════════════════


class TestAsyncJsonlWriterClose:
    """AsyncJsonlWriter close 行为"""

    def test_append_after_close_raises(self, tmp_path: Path):
        """close() 后 append() 应抛出 RuntimeError"""
        writer = AsyncJsonlWriter(tmp_path / "closed.jsonl", buffer_size=10)
        writer.close()

        with pytest.raises(RuntimeError, match="writer already closed"):
            writer.append({"tick": 999})

    def test_close_idempotent(self, tmp_path: Path):
        """close() 可以多次调用不崩溃"""
        writer = AsyncJsonlWriter(tmp_path / "multi_close.jsonl", buffer_size=10)
        writer.close()
        writer.close()
        writer.close()
        # 不应抛出任何异常

    def test_close_drains_buffer(self, tmp_path: Path):
        """close() 等待数据排空"""
        output = tmp_path / "drain.jsonl"
        writer = AsyncJsonlWriter(output, buffer_size=100)

        for i in range(20):
            writer.append({"tick": i})

        writer.close()

        # 验证数据已写入文件
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 20, f"应写入 20 条记录，实际 {len(lines)}"


# ══════════════════════════════════════════════════════════
# 3. Network.check_death() min_neurons 边界测试
# ══════════════════════════════════════════════════════════


class TestCheckDeathMinNeurons:
    """Network.check_death() 的 min_neurons 保护机制"""

    def test_alive_not_below_min_neurons(self, ring_network_factory):
        """存活数不应低于 min_neurons"""
        min_n = 3
        net = ring_network_factory(n=6, min_neurons=min_n)

        # 让大部分神经元能量耗尽
        for i, neuron in enumerate(net.neurons.values()):
            if i < 5:
                neuron.energy = 0.0
                neuron.alive = True  # 仍标记为存活，check_death 会检测 energy<=0

        net.check_death()

        alive_count = sum(1 for n in net.neurons.values() if n.alive)
        assert alive_count >= min_n, (
            f"存活数 {alive_count} 不应低于 min_neurons={min_n}"
        )

    def test_rescued_neuron_energy(self, ring_network_factory):
        """被救回的神经元 energy 应 >= RESCUED_NEURON_ENERGY"""
        min_n = 3
        net = ring_network_factory(n=6, min_neurons=min_n)

        # 让所有神经元能量耗尽
        for neuron in net.neurons.values():
            neuron.energy = 0.0

        net.check_death()

        alive_neurons = [n for n in net.neurons.values() if n.alive]
        assert len(alive_neurons) >= min_n

        for neuron in alive_neurons:
            assert neuron.energy >= RESCUED_NEURON_ENERGY, (
                f"被救回的神经元 energy={neuron.energy} "
                f"应 >= RESCUED_NEURON_ENERGY={RESCUED_NEURON_ENERGY}"
            )

    def test_min_neurons_zero_allows_all_death(self, ring_network_factory):
        """min_neurons=0 时允许全部死亡"""
        net = ring_network_factory(n=4, min_neurons=0)

        for neuron in net.neurons.values():
            neuron.energy = 0.0

        net.check_death()
        alive_count = sum(1 for n in net.neurons.values() if n.alive)
        assert alive_count == 0, "min_neurons=0 时应允许全部死亡"


# ══════════════════════════════════════════════════════════
# 4. RuleStore LRU 淘汰测试
# ══════════════════════════════════════════════════════════


class TestRuleStoreLRU:
    """RuleStore LRU 淘汰策略"""

    def test_touch_trigram_moves_to_end(self):
        """_touch_trigram 应将规则移到末尾"""
        trigrams = OrderedDict([
            ((1, 2, 3), 10),
            ((4, 5, 6), 20),
            ((7, 8, 9), 30),
        ])
        store = RuleStore(trigram_support=trigrams, bigram_support={})

        # 访问第一个规则
        store._touch_trigram((1, 2, 3))

        # 验证 (1,2,3) 移到了末尾
        keys = list(store.trigram_support.keys())
        assert keys[-1] == (1, 2, 3), (
            f"(1,2,3) 应在末尾，实际顺序: {keys}"
        )
        assert keys[0] == (4, 5, 6), "原第二个应变为第一个"

    def test_prune_rules_evicts_from_head(self):
        """_prune_rules 应从头部（最久未访问）淘汰"""
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            max_rules=4,  # tri_cap=2, bi_cap=2
        )
        trigrams = OrderedDict([
            ((1, 2, 3), 10),
            ((4, 5, 6), 20),
            ((7, 8, 9), 30),
        ])
        auto.store = RuleStore(trigram_support=trigrams, bigram_support={})
        auto.rule_count = 3

        auto._prune_rules()

        # tri_cap = 4 // 2 = 2，应淘汰 1 个（从头部）
        assert len(auto.store.trigram_support) <= 2
        # 头部 (1,2,3) 最久未访问，应被淘汰
        assert (1, 2, 3) not in auto.store.trigram_support
        # 末尾的应被保留
        assert (7, 8, 9) in auto.store.trigram_support

    def test_accessed_rules_not_evicted(self):
        """访问过的规则不应被淘汰"""
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            max_rules=4,  # tri_cap=2
        )
        trigrams = OrderedDict([
            ((1, 2, 3), 10),
            ((4, 5, 6), 20),
            ((7, 8, 9), 30),
        ])
        auto.store = RuleStore(trigram_support=trigrams, bigram_support={})
        auto.rule_count = 3

        # 访问 (1,2,3) 使其移到末尾
        auto.store.trigram_confidence((1, 2, 3))

        auto._prune_rules()

        # (1,2,3) 最近被访问，应被保留
        assert (1, 2, 3) in auto.store.trigram_support
        # (4,5,6) 最久未访问，应被淘汰
        assert (4, 5, 6) not in auto.store.trigram_support

    def test_bigram_lru_eviction(self):
        """bigram 规则也应遵循 LRU 淘汰"""
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            max_rules=4,  # bi_cap=2
        )
        bigrams = OrderedDict([
            ((10, 20), 5),
            ((30, 40), 15),
            ((50, 60), 25),
        ])
        auto.store = RuleStore(trigram_support={}, bigram_support=bigrams)
        auto.rule_count = 3

        auto._prune_rules()

        assert len(auto.store.bigram_support) <= 2
        # 头部 (10,20) 应被淘汰
        assert (10, 20) not in auto.store.bigram_support
        assert (50, 60) in auto.store.bigram_support


# ══════════════════════════════════════════════════════════
# 5. visualization.py matplotlib 异常保护测试
# ══════════════════════════════════════════════════════════


class TestLiveMonitorExceptionProtection:
    """LiveMonitor.on_tick 异常保护"""

    def test_on_tick_no_crash_on_matplotlib_error(self):
        """matplotlib 操作抛出异常时 on_tick 不崩溃"""
        import matplotlib
        matplotlib.use("Agg")

        from neurogenesis_lab.visualization import LiveMonitor, MATPLOTLIB_AVAILABLE

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib 未安装")

        monitor = LiveMonitor(plot_interval=1)

        snapshot = {
            "alive_neurons": 10,
            "total_neurons": 15,
            "mean_energy": 1.0,
            "prediction_error": 0.5,
            "rule_count": 0,
        }

        # 第一次调用初始化图表
        monitor.on_tick(1, snapshot)

        # 用 mock 让 _update_plots 抛出异常
        with patch.object(
            monitor, "_update_plots", side_effect=RuntimeError("渲染失败")
        ):
            # 不应崩溃
            monitor.on_tick(2, snapshot)

    def test_on_tick_continues_after_exception(self):
        """异常后后续 on_tick 仍可正常工作"""
        import matplotlib
        matplotlib.use("Agg")

        from neurogenesis_lab.visualization import LiveMonitor, MATPLOTLIB_AVAILABLE

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib 未安装")

        monitor = LiveMonitor(plot_interval=1)

        snapshot = {
            "alive_neurons": 10,
            "total_neurons": 15,
            "mean_energy": 1.0,
            "prediction_error": 0.5,
            "rule_count": 0,
        }

        # 第一次正常
        monitor.on_tick(1, snapshot)

        # 第二次异常
        with patch.object(
            monitor, "_update_plots", side_effect=RuntimeError("渲染失败")
        ):
            monitor.on_tick(2, snapshot)

        # 第三次恢复正常 — 不应崩溃
        monitor.on_tick(3, snapshot)

        # 数据应正常累积
        assert len(monitor._ticks) == 3

    def test_plot_functions_return_none_on_error(self):
        """绘图函数在异常时返回 None 而非崩溃"""
        import matplotlib
        matplotlib.use("Agg")

        from neurogenesis_lab.visualization import (
            plot_population_curve,
            MATPLOTLIB_AVAILABLE,
        )

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib 未安装")

        # 传入格式错误的数据，触发内部异常
        bad_data = [{"metrics": {"tick": "not_a_number"}}]
        result = plot_population_curve(bad_data, show=False)
        # 应返回 None 而非崩溃
        assert result is None
