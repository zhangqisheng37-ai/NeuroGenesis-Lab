"""网络生命周期演化 — network 子包的繁殖、死亡与尸体回收部分。"""

from __future__ import annotations

import logging
import math
import random

from neurogenesis_lab.config import (
    CONGESTION_LINK_THRESHOLD,
    RESCUED_NEURON_ENERGY,
    SPAWN_LINK_DELAY,
    SPAWN_LINK_WEIGHT,
)
from neurogenesis_lab.evolution import (
    can_reproduce,
    decay_corpse_energy,
    is_corpse_expired,
    select_rescued,
)
from neurogenesis_lab.network._base import _NetworkAttrs
from neurogenesis_lab.network.corpse import Corpse
from neurogenesis_lab.neuron import Neuron

_logger = logging.getLogger(__name__)


class NetworkLifecycleMixin(_NetworkAttrs):
    """繁殖检查、死亡检查、尸体更新与连接修剪。"""

    def _run_evolution_step(self) -> tuple[int, int]:
        self._update_corpses()
        births = self.check_reproduction()
        deaths = self.check_death()
        self.birth_count += births
        self.death_count += deaths
        return births, deaths

    def check_reproduction(self) -> int:
        """执行繁殖检查 — 高适应度且能量充足的神经元产生子代。

        Returns:
            本轮新增的子代数量。
        """
        if len(self.neurons) >= self.max_neurons:
            return 0
        candidates = [neuron for neuron in self.neurons.values() if neuron.alive]
        # 按 fitness 排序 — 高适应度优先繁殖
        candidates.sort(key=lambda n: n.fitness(self.fitness_weights), reverse=True)

        # 计算自适应繁殖成本
        alive_count = len(candidates)
        effective_share = self._calculate_adaptive_reproduction_share(alive_count)
        effective_mutation_scale = (
            self.evolution_mutation_amplitude
            if self.evolution_mutation_amplitude is not None
            else self.mutation_scale
        )
        births = 0

        for neuron in candidates:
            if len(self.neurons) >= self.max_neurons:
                break
            if not can_reproduce(neuron.energy, self.reproduction_threshold):
                continue
            # 拥塞惩罚: 该神经元连接过多时降低繁殖概率
            link_count = len(neuron.links)
            if link_count > CONGESTION_LINK_THRESHOLD and random.random() < self.congestion_penalty:
                continue
            child = neuron.spawn(max(0.0, effective_mutation_scale), effective_share)
            neuron.energy = max(0.0, neuron.energy - child.energy)
            self.add_neuron(child)
            self.connect(neuron.id, child.id, weight=SPAWN_LINK_WEIGHT, delay=SPAWN_LINK_DELAY)
            self.connect(child.id, neuron.id, weight=SPAWN_LINK_WEIGHT, delay=SPAWN_LINK_DELAY)
            births += 1
        self._prune_links()
        return births

    def check_death(self) -> int:
        """执行死亡检查 — 移除能量耗尽的神经元，生成尸体。

        当存活数低于 min_neurons 时启动保护机制，复活能量最高的死亡神经元。

        Returns:
            本轮死亡的神经元数量。
        """
        if len(self.neurons) > self.max_neurons:
            ordered = sorted(self.neurons.values(), key=lambda item: item.energy)
            overflow = len(self.neurons) - self.max_neurons
            for neuron in ordered[:overflow]:
                neuron.alive = False

        dead_ids = [
            neuron.id
            for neuron in self.neurons.values()
            if not neuron.alive or neuron.energy <= 0.0
        ]
        # 保护 min_neurons 下限
        alive_count = len(self.neurons) - len(dead_ids)
        if alive_count < self.min_neurons:
            _logger.warning(
                "种群触及 min_neurons 下限: alive=%d, dead=%d, min=%d — 启动保护机制",
                alive_count,
                len(dead_ids),
                self.min_neurons,
            )
            # 保留能量最高的死亡神经元
            save_count = self.min_neurons - alive_count
            candidates = [
                (nid, self.neurons[nid].energy) for nid in dead_ids if nid in self.neurons
            ]
            saved = set(select_rescued(candidates, save_count))
            for nid in saved:
                self.neurons[nid].alive = True
                self.neurons[nid].energy = max(RESCUED_NEURON_ENERGY, self.neurons[nid].energy)
            dead_ids = [nid for nid in dead_ids if nid not in saved]

        deaths = 0
        for neuron_id in dead_ids:
            neuron_obj: Neuron | None = self.neurons.get(neuron_id)
            if neuron_obj is None:
                continue
            params = {
                "tau_m": neuron_obj.tau_m,
                "r_m": neuron_obj.r_m,
                "v_th": neuron_obj.v_th,
                "v_rest": neuron_obj.v_rest,
                "v_reset": neuron_obj.v_reset,
                "spike_cost": neuron_obj.spike_cost,
                "input_gain": neuron_obj.input_gain,
            }
            for attr in ["a", "b", "c", "d", "v_peak", "delta_t", "tau_w"]:
                if hasattr(neuron_obj, attr):
                    value = getattr(neuron_obj, attr)
                    if isinstance(value, (int, float)):
                        params[attr] = float(value)
            self.corpses.append(
                self.corpse_pool.acquire(
                    pos=neuron_obj.pos,
                    energy=neuron_obj.energy,
                    params=params,
                    ttl=self.corpse_ttl,
                )
            )
            self.neurons.pop(neuron_id, None)
            deaths += 1
        return deaths

    def _update_corpses(self) -> None:
        if not self.corpses:
            return
        alive = [neuron for neuron in self.neurons.values() if neuron.alive]
        updated: list[Corpse] = []

        # 构建简单网格索引加速近邻搜索
        cell_size = self.scavenger_radius if self.scavenger_radius > 0 else 1.0
        grid: dict[tuple[int, int], list[Neuron]] = {}
        for n in alive:
            key = (int(n.pos[0] // cell_size), int(n.pos[1] // cell_size))
            grid.setdefault(key, []).append(n)

        for corpse in self.corpses:
            corpse.ttl -= 1
            # 能量衰减
            corpse.energy = decay_corpse_energy(corpse.energy, corpse.decay_rate)
            if corpse.energy > 0 and alive:
                cx, cy = int(corpse.pos[0] // cell_size), int(corpse.pos[1] // cell_size)
                nearest = None
                nearest_dist = None
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for n in grid.get((cx + dx, cy + dy), []):
                            ddx = n.pos[0] - corpse.pos[0]
                            ddy = n.pos[1] - corpse.pos[1]
                            dist = math.sqrt(ddx * ddx + ddy * ddy)
                            if dist <= self.scavenger_radius and (
                                nearest_dist is None or dist < nearest_dist
                            ):
                                nearest, nearest_dist = n, dist
                if nearest is not None:
                    nearest.energy += corpse.energy
                    corpse.energy = 0.0
            if is_corpse_expired(corpse.ttl, corpse.energy):
                self.corpse_pool.release(corpse, self.corpse_pool_max_size)
            else:
                updated.append(corpse)
        self.corpses = updated

    def _prune_links(self) -> None:
        total_links = sum(len(neuron.links) for neuron in self.neurons.values())
        if total_links <= self.max_links:
            return
        all_links: list[tuple[str, str]] = []
        for neuron_id, neuron in self.neurons.items():
            for target_id in neuron.links.keys():
                all_links.append((neuron_id, target_id))
        random.shuffle(all_links)
        to_remove = total_links - self.max_links
        for source_id, target_id in all_links[:to_remove]:
            if source_id in self.neurons:
                self.neurons[source_id].links.pop(target_id, None)
