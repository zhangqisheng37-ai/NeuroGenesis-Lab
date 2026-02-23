"""P3 常量正确性测试"""

from neurogenesis_lab.config import (
    DECAY_HIGH_MULTIPLIER,
    DECAY_LOW_MULTIPLIER,
    DENSITY_HIGH_THRESHOLD,
    DENSITY_LOW_THRESHOLD,
    HERITABLE_ATTRS,
    HERITABLE_DYNAMICS_ATTRS,
    HERITABLE_SUBCLASS_ATTRS,
    MOVE_JITTER_RANGE,
    POPULATION_HIGH_THRESHOLD,
    POPULATION_LOW_THRESHOLD,
    REPRO_SHARE_HIGH_MULTIPLIER,
    REPRO_SHARE_LOW_MULTIPLIER,
    REPRO_SHARE_MAX,
    REPRO_SHARE_MIN,
    RESCUED_NEURON_ENERGY,
    SPAWN_POSITION_JITTER,
)


class TestConstantValues:
    """常量值正确性"""

    def test_density_thresholds(self) -> None:
        assert DENSITY_HIGH_THRESHOLD == 0.8
        assert DENSITY_LOW_THRESHOLD == 0.3

    def test_decay_multipliers(self) -> None:
        assert DECAY_HIGH_MULTIPLIER == 1.2
        assert DECAY_LOW_MULTIPLIER == 0.7

    def test_population_thresholds(self) -> None:
        assert POPULATION_HIGH_THRESHOLD == 0.8
        assert POPULATION_LOW_THRESHOLD == 0.3

    def test_repro_share_multipliers(self) -> None:
        assert REPRO_SHARE_HIGH_MULTIPLIER == 1.3
        assert REPRO_SHARE_LOW_MULTIPLIER == 0.7

    def test_repro_share_bounds(self) -> None:
        assert REPRO_SHARE_MAX == 0.6
        assert REPRO_SHARE_MIN == 0.2

    def test_rescued_neuron_energy(self) -> None:
        assert RESCUED_NEURON_ENERGY == 0.1

    def test_spawn_position_jitter(self) -> None:
        assert SPAWN_POSITION_JITTER == 0.5

    def test_move_jitter_range(self) -> None:
        assert MOVE_JITTER_RANGE == 1.0


class TestHeritableAttrs:
    """遗传属性常量完整性"""

    def test_dynamics_attrs_content(self) -> None:
        expected = ("tau_m", "r_m", "v_th", "v_rest", "v_reset",
                    "spike_cost", "input_gain", "move_cost")
        assert HERITABLE_DYNAMICS_ATTRS == expected

    def test_subclass_attrs_content(self) -> None:
        expected = ("a", "b", "c", "d", "v_peak", "delta_t", "tau_w")
        assert HERITABLE_SUBCLASS_ATTRS == expected

    def test_heritable_attrs_is_concatenation(self) -> None:
        assert HERITABLE_ATTRS == HERITABLE_DYNAMICS_ATTRS + HERITABLE_SUBCLASS_ATTRS

    def test_no_duplicates(self) -> None:
        assert len(set(HERITABLE_ATTRS)) == len(HERITABLE_ATTRS)


class TestAdaptiveDecayUsesConstants:
    """自适应衰减/繁殖使用常量验证"""

    def test_adaptive_decay_high_density(self) -> None:
        from neurogenesis_lab.neuron import LIFNeuron
        from neurogenesis_lab.network import Network

        neurons = {f"n{i}": LIFNeuron(energy=2.0) for i in range(45)}
        net = Network(neurons=neurons, max_neurons=50, adaptive_decay=True)
        # 45/50 = 0.9 > DENSITY_HIGH_THRESHOLD
        result = net._calculate_adaptive_decay(45)
        assert result == net.energy_decay * DECAY_HIGH_MULTIPLIER

    def test_adaptive_decay_low_density(self) -> None:
        from neurogenesis_lab.neuron import LIFNeuron
        from neurogenesis_lab.network import Network

        neurons = {f"n{i}": LIFNeuron(energy=2.0) for i in range(5)}
        net = Network(neurons=neurons, max_neurons=50, adaptive_decay=True)
        # 5/50 = 0.1 < DENSITY_LOW_THRESHOLD
        result = net._calculate_adaptive_decay(5)
        assert result == net.energy_decay * DECAY_LOW_MULTIPLIER

    def test_adaptive_reproduction_high_population(self) -> None:
        from neurogenesis_lab.neuron import LIFNeuron
        from neurogenesis_lab.network import Network

        neurons = {f"n{i}": LIFNeuron(energy=2.0) for i in range(45)}
        net = Network(neurons=neurons, max_neurons=50, adaptive_reproduction=True)
        result = net._calculate_adaptive_reproduction_share(45)
        expected = min(REPRO_SHARE_MAX, net.reproduction_share * REPRO_SHARE_HIGH_MULTIPLIER)
        assert result == expected

    def test_adaptive_reproduction_low_population(self) -> None:
        from neurogenesis_lab.neuron import LIFNeuron
        from neurogenesis_lab.network import Network

        neurons = {f"n{i}": LIFNeuron(energy=2.0) for i in range(5)}
        net = Network(neurons=neurons, max_neurons=50, adaptive_reproduction=True)
        result = net._calculate_adaptive_reproduction_share(5)
        expected = max(REPRO_SHARE_MIN, net.reproduction_share * REPRO_SHARE_LOW_MULTIPLIER)
        assert result == expected
