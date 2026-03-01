"""单元测试 — CLI 配置映射"""

from argparse import Namespace
from pathlib import Path

import pytest

from neurogenesis_lab.__main__ import _build_sim_config


def test_build_sim_config_missing_sections_uses_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "minimal.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = 120
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=None,
        seed=None,
        dt=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )
    cfg = _build_sim_config(args)

    assert cfg.ticks == 120
    # 缺失 section 应回退默认值
    assert cfg.neuron.v_rest == -65.0
    assert cfg.network.max_neurons == 50
    assert cfg.evolution.min_neurons == 2


def test_build_sim_config_nested_override_partial_fitness_weights(tmp_path: Path) -> None:
    cfg_file = tmp_path / "fitness.toml"
    cfg_file.write_text(
        """
[evolution.fitness_weights]
propagation = 1.8
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=None,
        seed=None,
        dt=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )
    cfg = _build_sim_config(args)

    assert cfg.evolution.fitness_weights["propagation"] == 1.8
    # 其余键应保留默认值
    assert cfg.evolution.fitness_weights["energy_efficiency"] == 0.5
    assert cfg.evolution.fitness_weights["prediction_hit"] == 0.8


def test_build_sim_config_invalid_numeric_value_raises(tmp_path: Path) -> None:
    cfg_file = tmp_path / "invalid.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = "abc"
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=None,
        seed=None,
        dt=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )

    with pytest.raises((TypeError, ValueError)):
        _build_sim_config(args)


def test_build_sim_config_maps_all_toml_sections(tmp_path: Path) -> None:
    cfg_file = tmp_path / "experiment.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = 222
seed = 99
dt = 0.5

[neuron]
v_rest = -66.0
v_reset = -64.0
v_th = -54.0
tau_m = 12.0
r_m = 1.2
spike_cost = 0.03
input_gain = 0.08
mem_size = 77

[network]
reproduction_threshold = 3.2
reproduction_share = 0.33
mutation_scale = 0.07
energy_decay = 0.004
max_neurons = 88
max_links = 333
corpse_ttl = 66
scavenger_radius = 3.3
input_current_scale = 2.2
background_current = 0.6
spontaneous_spike_rate = 0.12
low_activation_threshold = 3
move_cost = 0.02

[evolution]
mutation_amplitude = 0.09
energy_budget_per_tick = 1.5
congestion_penalty = 0.21
min_neurons = 0

[evolution.fitness_weights]
propagation = 1.1
energy_efficiency = 0.7
prediction_hit = 0.9
""",
        encoding="utf-8",
    )

    args = Namespace(config=str(cfg_file), ticks=None, seed=None, dt=None)
    cfg = _build_sim_config(args)

    assert cfg.ticks == 222
    assert cfg.seed == 99
    assert cfg.dt == 0.5

    assert cfg.neuron.v_rest == -66.0
    assert cfg.neuron.v_reset == -64.0
    assert cfg.neuron.v_th == -54.0
    assert cfg.neuron.tau_m == 12.0
    assert cfg.neuron.r_m == 1.2
    assert cfg.neuron.spike_cost == 0.03
    assert cfg.neuron.input_gain == 0.08
    assert cfg.neuron.mem_size == 77

    assert cfg.network.reproduction_threshold == 3.2
    assert cfg.network.reproduction_share == 0.33
    assert cfg.network.mutation_scale == 0.07
    assert cfg.network.energy_decay == 0.004
    assert cfg.network.max_neurons == 88
    assert cfg.network.max_links == 333
    assert cfg.network.corpse_ttl == 66
    assert cfg.network.scavenger_radius == 3.3
    assert cfg.network.input_current_scale == 2.2
    assert cfg.network.background_current == 0.6
    assert cfg.network.spontaneous_spike_rate == 0.12
    assert cfg.network.low_activation_threshold == 3
    assert cfg.network.move_cost == 0.02

    assert cfg.evolution.mutation_amplitude == 0.09
    assert cfg.evolution.energy_budget_per_tick == 1.5
    assert cfg.evolution.congestion_penalty == 0.21
    assert cfg.evolution.min_neurons == 0
    assert cfg.evolution.fitness_weights["propagation"] == 1.1
    assert cfg.evolution.fitness_weights["energy_efficiency"] == 0.7
    assert cfg.evolution.fitness_weights["prediction_hit"] == 0.9


def test_build_sim_config_cli_overrides_toml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "experiment.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = 100
seed = 7
dt = 1.0
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=300,
        seed=123,
        dt=0.25,
        background_current=0.4,
        spontaneous_spike_rate=0.2,
        low_activation_threshold=5,
    )
    cfg = _build_sim_config(args)

    assert cfg.ticks == 300
    assert cfg.seed == 123
    assert cfg.dt == 0.25
    assert cfg.network.background_current == 0.4
    assert cfg.network.spontaneous_spike_rate == 0.2
    assert cfg.network.low_activation_threshold == 5
