"""P1.2 Network 配置/状态分离测试"""

import pickle
import tempfile
from pathlib import Path

from neurogenesis_lab.network import Network, NetworkConfig, NetworkState
from neurogenesis_lab.neuron import LIFNeuron


class TestConfigStateAccessibility:
    """config/state 组件可访问性"""

    def test_config_fields_accessible(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100, energy_decay=0.01)
        assert net._config.max_neurons == net.max_neurons == 100
        assert net._config.energy_decay == net.energy_decay == 0.01

    def test_state_fields_accessible(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n})
        net.total_spikes = 42
        assert net._state.total_spikes == net.total_spikes == 42

    def test_all_config_fields_routed(self) -> None:
        n = LIFNeuron()
        net = Network(
            neurons={n.id: n},
            reproduction_threshold=3.0,
            reproduction_share=0.5,
            mutation_scale=0.1,
            energy_decay=0.01,
            max_neurons=100,
            max_links=300,
            corpse_ttl=30,
            scavenger_radius=3.0,
            input_current_scale=2.0,
            background_current=1.0,
            spontaneous_spike_rate=0.05,
            move_cost=0.02,
            energy_cap=10.0,
            adaptive_decay=False,
            adaptive_reproduction=False,
            endangered_threshold=5,
            congestion_penalty=0.2,
            min_neurons=3,
            energy_budget_per_tick=1.0,
        )
        assert net.reproduction_threshold == 3.0
        assert net.reproduction_share == 0.5
        assert net.mutation_scale == 0.1
        assert net.energy_decay == 0.01
        assert net.max_neurons == 100
        assert net.max_links == 300
        assert net.corpse_ttl == 30
        assert net.scavenger_radius == 3.0
        assert net.input_current_scale == 2.0
        assert net.background_current == 1.0
        assert net.spontaneous_spike_rate == 0.05
        assert net.move_cost == 0.02
        assert net.energy_cap == 10.0
        assert net.adaptive_decay is False
        assert net.adaptive_reproduction is False
        assert net.endangered_threshold == 5
        assert net.congestion_penalty == 0.2
        assert net.min_neurons == 3
        assert net.energy_budget_per_tick == 1.0


class TestSetAttrRouting:
    """setattr 路由正确性"""

    def test_set_config_field(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        net.max_neurons = 200
        assert net._config.max_neurons == 200
        assert net.max_neurons == 200

    def test_set_state_field(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        net.birth_count = 10
        assert net._state.birth_count == 10
        assert net.birth_count == 10

    def test_set_direct_field(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        net.expected_state = {"energy": 1.0}
        assert net.expected_state == {"energy": 1.0}
        assert "expected_state" in net.__dict__


class TestPickleRoundTrip:
    """pickle 新旧格式往返"""

    def test_pickle_roundtrip(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100)
        net.total_spikes = 42
        data = pickle.dumps(net)
        restored = pickle.loads(data)
        assert restored.max_neurons == 100
        assert restored.total_spikes == 42
        assert hasattr(restored, "_config")
        assert hasattr(restored, "_state")

    def test_checkpoint_roundtrip(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100)
        net.tick(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt"
            net.save_checkpoint(path)
            restored = Network.load_checkpoint(path)
            assert restored.max_neurons == 100
            assert hasattr(restored, "_config")
            assert hasattr(restored, "_state")

    def test_old_format_pickle_compat(self) -> None:
        """模拟旧格式 pickle（无 _config/_state）"""
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100)
        # 手动构造旧格式 state
        old_state = dict(net.__dict__)
        config_copy = {
            k: getattr(net._config, k)
            for k in NetworkConfig.__dataclass_fields__
        }
        state_copy = {
            k: getattr(net._state, k)
            for k in NetworkState.__dataclass_fields__
        }
        old_state.pop("_config")
        old_state.pop("_state")
        old_state.update(config_copy)
        old_state.update(state_copy)

        restored = Network.__new__(Network)
        restored.__setstate__(old_state)
        assert restored.max_neurons == 100
        assert hasattr(restored, "_config")
        assert hasattr(restored, "_state")


class TestConstructionSignature:
    """构造签名不变验证"""

    def test_keyword_construction(self) -> None:
        n = LIFNeuron()
        net = Network(
            neurons={n.id: n},
            reproduction_threshold=3.0,
            max_neurons=100,
            birth_count=5,
        )
        assert net.reproduction_threshold == 3.0
        assert net.max_neurons == 100
        assert net.birth_count == 5

    def test_direct_fields_remain(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        # 这些字段应直接在 __dict__ 中
        assert "neurons" in net.__dict__
        assert "queue" in net.__dict__
        assert "replay" in net.__dict__
        assert "corpses" in net.__dict__
        assert "corpse_pool" in net.__dict__

    def test_config_state_fields_not_in_dict(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        # 配置/状态字段不应在 __dict__ 中
        assert "max_neurons" not in net.__dict__
        assert "birth_count" not in net.__dict__
        assert "total_spikes" not in net.__dict__
