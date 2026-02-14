"""集中配置系统 — 参数可控化 + 随机种子统一入口"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class Config:
    """向后兼容的最小配置"""
    ticks: int = 10
    network_size: int = 16
    energy_decay: float = 0.01
    seed: int = 42


def default_config() -> Config:
    return Config()


@dataclass
class NeuronParams:
    """神经元动力学参数"""
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_th: float = -55.0
    tau_m: float = 20.0
    r_m: float = 1.0
    spike_cost: float = 0.05
    input_gain: float = 0.05
    mem_size: int = 64

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.v_rest):
            errors.append(f"v_rest 不是有限数: {self.v_rest}")
        if not math.isfinite(self.v_reset):
            errors.append(f"v_reset 不是有限数: {self.v_reset}")
        if not math.isfinite(self.v_th):
            errors.append(f"v_th 不是有限数: {self.v_th}")
        if self.tau_m <= 0:
            errors.append(f"tau_m 必须为正: {self.tau_m}")
        if self.spike_cost < 0:
            errors.append(f"spike_cost 不能为负: {self.spike_cost}")
        if self.input_gain < 0:
            errors.append(f"input_gain 不能为负: {self.input_gain}")
        if self.mem_size <= 0:
            errors.append(f"mem_size 必须为正: {self.mem_size}")
        return errors


@dataclass
class NetworkParams:
    """网络级参数"""
    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    energy_decay: float = 0.003
    max_neurons: int = 50
    max_links: int = 200
    corpse_ttl: int = 50
    scavenger_radius: float = 2.5
    input_current_scale: float = 1.5
    move_cost: float = 0.01

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.reproduction_threshold <= 0:
            errors.append(f"reproduction_threshold 必须为正: {self.reproduction_threshold}")
        if not (0 < self.reproduction_share <= 1):
            errors.append(f"reproduction_share 必须在 (0, 1]: {self.reproduction_share}")
        if self.mutation_scale < 0:
            errors.append(f"mutation_scale 不能为负: {self.mutation_scale}")
        if self.energy_decay < 0:
            errors.append(f"energy_decay 不能为负: {self.energy_decay}")
        if self.max_neurons <= 0:
            errors.append(f"max_neurons 必须为正: {self.max_neurons}")
        if self.max_links <= 0:
            errors.append(f"max_links 必须为正: {self.max_links}")
        if self.corpse_ttl <= 0:
            errors.append(f"corpse_ttl 必须为正: {self.corpse_ttl}")
        if self.scavenger_radius < 0:
            errors.append(f"scavenger_radius 不能为负: {self.scavenger_radius}")
        if self.input_current_scale < 0:
            errors.append(f"input_current_scale 不能为负: {self.input_current_scale}")
        if self.move_cost < 0:
            errors.append(f"move_cost 不能为负: {self.move_cost}")
        return errors


@dataclass
class EvolutionParams:
    """演化系统参数"""
    fitness_weights: dict[str, float] = field(default_factory=lambda: {
        "propagation": 1.0,
        "energy_efficiency": 0.5,
        "prediction_hit": 0.8,
    })
    mutation_amplitude: float = 0.05
    energy_budget_per_tick: float = 0.0  # 0 = 不限
    congestion_penalty: float = 0.1
    min_neurons: int = 2

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.mutation_amplitude < 0:
            errors.append(f"mutation_amplitude 不能为负: {self.mutation_amplitude}")
        if self.min_neurons < 0:
            errors.append(f"min_neurons 不能为负: {self.min_neurons}")
        return errors


@dataclass
class SimConfig:
    """仿真总配置 — 统一入口"""
    ticks: int = 100
    seed: int = 42
    dt: float = 1.0
    neuron: NeuronParams = field(default_factory=NeuronParams)
    network: NetworkParams = field(default_factory=NetworkParams)
    evolution: EvolutionParams = field(default_factory=EvolutionParams)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ticks <= 0:
            errors.append(f"ticks 必须为正: {self.ticks}")
        if self.dt <= 0:
            errors.append(f"dt 必须为正: {self.dt}")
        if not math.isfinite(self.dt):
            errors.append(f"dt 不是有限数: {self.dt}")
        errors.extend(self.neuron.validate())
        errors.extend(self.network.validate())
        errors.extend(self.evolution.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 jsonl 记录）"""
        return {
            "schema_version": self.schema_version,
            "ticks": self.ticks,
            "seed": self.seed,
            "dt": self.dt,
            "neuron": {
                "v_rest": self.neuron.v_rest,
                "v_reset": self.neuron.v_reset,
                "v_th": self.neuron.v_th,
                "tau_m": self.neuron.tau_m,
                "r_m": self.neuron.r_m,
                "spike_cost": self.neuron.spike_cost,
                "input_gain": self.neuron.input_gain,
                "mem_size": self.neuron.mem_size,
            },
            "network": {
                "reproduction_threshold": self.network.reproduction_threshold,
                "reproduction_share": self.network.reproduction_share,
                "mutation_scale": self.network.mutation_scale,
                "energy_decay": self.network.energy_decay,
                "max_neurons": self.network.max_neurons,
                "max_links": self.network.max_links,
                "corpse_ttl": self.network.corpse_ttl,
                "scavenger_radius": self.network.scavenger_radius,
                "input_current_scale": self.network.input_current_scale,
                "move_cost": self.network.move_cost,
            },
            "evolution": {
                "fitness_weights": self.evolution.fitness_weights,
                "mutation_amplitude": self.evolution.mutation_amplitude,
                "energy_budget_per_tick": self.evolution.energy_budget_per_tick,
                "congestion_penalty": self.evolution.congestion_penalty,
                "min_neurons": self.evolution.min_neurons,
            },
        }


def default_sim_config() -> SimConfig:
    return SimConfig()


def seed_all(seed: int) -> None:
    """统一随机种子入口 — 覆盖所有 random 调用路径"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
