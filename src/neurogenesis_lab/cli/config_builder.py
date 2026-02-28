"""CLI 配置构建 — TOML 加载与 SimConfig 组装"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurogenesis_lab.config import (
    EvolutionParams,
    NetworkParams,
    NeuronParams,
    SimConfig,
)


def load_toml_config(path: Path) -> dict[str, object]:
    """加载 TOML 配置文件。

    Args:
        path: TOML 文件路径。

    Returns:
        解析后的字典。
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    """安全获取 TOML 配置文件中的节。"""
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _float_map(values: dict[str, object], allowed: set[str]) -> dict[str, float]:
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
            ticks=int(sim.get("ticks", cfg.ticks)),  # type: ignore[arg-type]
            seed=int(sim.get("seed", cfg.seed)),  # type: ignore[arg-type]
            dt=float(sim.get("dt", cfg.dt)),  # type: ignore[arg-type]
            neuron=NeuronParams(
                v_rest=float(neuron.get("v_rest", cfg.neuron.v_rest)),  # type: ignore[arg-type]
                v_reset=float(neuron.get("v_reset", cfg.neuron.v_reset)),  # type: ignore[arg-type]
                v_th=float(neuron.get("v_th", cfg.neuron.v_th)),  # type: ignore[arg-type]
                tau_m=float(neuron.get("tau_m", cfg.neuron.tau_m)),  # type: ignore[arg-type]
                r_m=float(neuron.get("r_m", cfg.neuron.r_m)),  # type: ignore[arg-type]
                spike_cost=float(neuron.get("spike_cost", cfg.neuron.spike_cost)),  # type: ignore[arg-type]
                input_gain=float(neuron.get("input_gain", cfg.neuron.input_gain)),  # type: ignore[arg-type]
                mem_size=int(neuron.get("mem_size", cfg.neuron.mem_size)),  # type: ignore[arg-type]
            ),
            network=NetworkParams(
                reproduction_threshold=float(
                    network.get("reproduction_threshold", cfg.network.reproduction_threshold)  # type: ignore[arg-type]
                ),
                reproduction_share=float(
                    network.get("reproduction_share", cfg.network.reproduction_share)  # type: ignore[arg-type]
                ),
                mutation_scale=float(
                    network.get("mutation_scale", cfg.network.mutation_scale)  # type: ignore[arg-type]
                ),
                energy_decay=float(
                    network.get("energy_decay", cfg.network.energy_decay)  # type: ignore[arg-type]
                ),
                max_neurons=int(network.get("max_neurons", cfg.network.max_neurons)),  # type: ignore[arg-type]
                max_links=int(network.get("max_links", cfg.network.max_links)),  # type: ignore[arg-type]
                corpse_ttl=int(network.get("corpse_ttl", cfg.network.corpse_ttl)),  # type: ignore[arg-type]
                scavenger_radius=float(
                    network.get("scavenger_radius", cfg.network.scavenger_radius)  # type: ignore[arg-type]
                ),
                input_current_scale=float(
                    network.get("input_current_scale", cfg.network.input_current_scale)  # type: ignore[arg-type]
                ),
                background_current=float(
                    network.get("background_current", cfg.network.background_current)  # type: ignore[arg-type]
                ),
                spontaneous_spike_rate=float(
                    network.get("spontaneous_spike_rate", cfg.network.spontaneous_spike_rate)  # type: ignore[arg-type]
                ),
                low_activation_threshold=int(
                    network.get("low_activation_threshold", cfg.network.low_activation_threshold)  # type: ignore[arg-type]
                ),
                move_cost=float(network.get("move_cost", cfg.network.move_cost)),  # type: ignore[arg-type]
                energy_cap=float(network.get("energy_cap", cfg.network.energy_cap)),  # type: ignore[arg-type]
                adaptive_decay=bool(
                    network.get("adaptive_decay", cfg.network.adaptive_decay)
                ),
                adaptive_reproduction=bool(
                    network.get(
                        "adaptive_reproduction", cfg.network.adaptive_reproduction
                    )
                ),
                endangered_threshold=int(
                    network.get("endangered_threshold", cfg.network.endangered_threshold)  # type: ignore[arg-type]
                ),
            ),
            evolution=EvolutionParams(
                fitness_weights={
                    **cfg.evolution.fitness_weights,
                    **_float_map(
                        fitness_weights,  # type: ignore[arg-type]
                        allowed={"propagation", "energy_efficiency", "prediction_hit"},
                    ),
                },
                mutation_amplitude=float(
                    evolution.get("mutation_amplitude", cfg.evolution.mutation_amplitude)  # type: ignore[arg-type]
                ),
                energy_budget_per_tick=float(
                    evolution.get("energy_budget_per_tick", cfg.evolution.energy_budget_per_tick)  # type: ignore[arg-type]
                ),
                congestion_penalty=float(
                    evolution.get("congestion_penalty", cfg.evolution.congestion_penalty)  # type: ignore[arg-type]
                ),
                min_neurons=int(
                    evolution.get("min_neurons", cfg.evolution.min_neurons)  # type: ignore[arg-type]
                ),
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
