"""CLI 配置构建 — TOML 加载与 SimConfig 组装"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from neurogenesis_lab.config import (
    EvolutionParams,
    NetworkParams,
    NeuronParams,
    SimConfig,
)


def load_toml_config(path: Path) -> dict[str, Any]:
    """加载 TOML 配置文件。

    Args:
        path: TOML 文件路径。

    Returns:
        解析后的字典。
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return dict(tomllib.load(f))


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """安全获取 TOML 配置文件中的节。"""
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _as_int(values: dict[str, Any], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"配置项 {key} 必须是 int")
    return int(value)


def _as_float(values: dict[str, Any], key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"配置项 {key} 必须是 float")
    return float(value)


def _as_bool(values: dict[str, Any], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"配置项 {key} 必须是 bool")
    return value


def _float_map(values: dict[str, Any], allowed: set[str]) -> dict[str, float]:
    """从字典中提取指定键的 float 值。"""
    mapped: dict[str, float] = {}
    for key, val in values.items():
        if key in allowed and isinstance(val, (int, float)):
            mapped[key] = float(val)
    return mapped


def build_sim_config(args: argparse.Namespace) -> SimConfig:
    """从命令行参数和可选 TOML 文件构建配置。

    Args:
        args: 已解析的命令行参数。

    Returns:
        构建完成的 SimConfig 实例。
    """
    cfg = SimConfig()

    if hasattr(args, "config") and args.config:
        toml_data = load_toml_config(Path(args.config))
        sim = _section(toml_data, "simulation")
        neuron = _section(toml_data, "neuron")
        network = _section(toml_data, "network")
        evolution = _section(toml_data, "evolution")
        fitness_weights = _section(evolution, "fitness_weights")

        cfg = SimConfig(
            ticks=_as_int(sim, "ticks", cfg.ticks),
            seed=_as_int(sim, "seed", cfg.seed),
            dt=_as_float(sim, "dt", cfg.dt),
            neuron=NeuronParams(
                v_rest=_as_float(neuron, "v_rest", cfg.neuron.v_rest),
                v_reset=_as_float(neuron, "v_reset", cfg.neuron.v_reset),
                v_th=_as_float(neuron, "v_th", cfg.neuron.v_th),
                tau_m=_as_float(neuron, "tau_m", cfg.neuron.tau_m),
                r_m=_as_float(neuron, "r_m", cfg.neuron.r_m),
                spike_cost=_as_float(neuron, "spike_cost", cfg.neuron.spike_cost),
                input_gain=_as_float(neuron, "input_gain", cfg.neuron.input_gain),
                mem_size=_as_int(neuron, "mem_size", cfg.neuron.mem_size),
            ),
            network=NetworkParams(
                reproduction_threshold=_as_float(
                    network, "reproduction_threshold", cfg.network.reproduction_threshold
                ),
                reproduction_share=_as_float(
                    network, "reproduction_share", cfg.network.reproduction_share
                ),
                mutation_scale=_as_float(network, "mutation_scale", cfg.network.mutation_scale),
                energy_decay=_as_float(network, "energy_decay", cfg.network.energy_decay),
                max_neurons=_as_int(network, "max_neurons", cfg.network.max_neurons),
                max_links=_as_int(network, "max_links", cfg.network.max_links),
                corpse_ttl=_as_int(network, "corpse_ttl", cfg.network.corpse_ttl),
                scavenger_radius=_as_float(
                    network, "scavenger_radius", cfg.network.scavenger_radius
                ),
                input_current_scale=_as_float(
                    network, "input_current_scale", cfg.network.input_current_scale
                ),
                background_current=_as_float(
                    network, "background_current", cfg.network.background_current
                ),
                spontaneous_spike_rate=_as_float(
                    network, "spontaneous_spike_rate", cfg.network.spontaneous_spike_rate
                ),
                low_activation_threshold=_as_int(
                    network, "low_activation_threshold", cfg.network.low_activation_threshold
                ),
                move_cost=_as_float(network, "move_cost", cfg.network.move_cost),
                energy_cap=_as_float(network, "energy_cap", cfg.network.energy_cap),
                adaptive_decay=_as_bool(
                    network, "adaptive_decay", cfg.network.adaptive_decay
                ),
                adaptive_reproduction=_as_bool(
                    network,
                    "adaptive_reproduction",
                    cfg.network.adaptive_reproduction,
                ),
                endangered_threshold=_as_int(
                    network, "endangered_threshold", cfg.network.endangered_threshold
                ),
            ),
            evolution=EvolutionParams(
                fitness_weights={
                    **cfg.evolution.fitness_weights,
                    **_float_map(
                        fitness_weights,
                        allowed={"propagation", "energy_efficiency", "prediction_hit"},
                    ),
                },
                mutation_amplitude=_as_float(
                    evolution,
                    "mutation_amplitude",
                    cfg.evolution.mutation_amplitude,
                ),
                energy_budget_per_tick=_as_float(
                    evolution,
                    "energy_budget_per_tick",
                    cfg.evolution.energy_budget_per_tick,
                ),
                congestion_penalty=_as_float(
                    evolution,
                    "congestion_penalty",
                    cfg.evolution.congestion_penalty,
                ),
                min_neurons=_as_int(evolution, "min_neurons", cfg.evolution.min_neurons),
            ),
        )
    # 命令行覆盖
    if hasattr(args, "ticks") and args.ticks is not None:
        cfg.ticks = args.ticks
    if hasattr(args, "seed") and args.seed is not None:
        cfg.seed = args.seed
    if hasattr(args, "dt") and args.dt is not None:
        cfg.dt = args.dt
    if hasattr(args, "background_current") and args.background_current is not None:
        cfg.network.background_current = float(args.background_current)
    if (
        hasattr(args, "spontaneous_spike_rate")
        and args.spontaneous_spike_rate is not None
    ):
        cfg.network.spontaneous_spike_rate = float(args.spontaneous_spike_rate)
    if (
        hasattr(args, "low_activation_threshold")
        and args.low_activation_threshold is not None
    ):
        cfg.network.low_activation_threshold = int(args.low_activation_threshold)
    return cfg


def validate_run_config(cfg: SimConfig) -> None:
    """验证运行配置合法性，失败时退出进程。

    Args:
        cfg: 待验证的配置实例。
    """
    errors = cfg.validate()
    if not errors:
        return
    print("配置验证失败:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
