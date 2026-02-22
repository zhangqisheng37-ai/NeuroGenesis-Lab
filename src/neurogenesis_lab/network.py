from __future__ import annotations

import heapq
import math
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import Neuron, calculate_reward, safe_float
from neurogenesis_lab.protocols import TickSnapshot, EngineSummary


@dataclass
class EventQueue:
    _heap: list[tuple[int, int, Event]] = field(default_factory=list)
    _seq: int = 0

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.t, self._seq, event))
        self._seq += 1

    def push_many(self, events: Iterable[Event]) -> None:
        for event in events:
            self.push(event)

    def pop_ready(self, tick: int) -> list[Event]:
        ready: list[Event] = []
        while self._heap and self._heap[0][0] <= tick:
            _, _, event = heapq.heappop(self._heap)
            ready.append(event)
        return ready

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class Corpse:
    pos: tuple[float, float]
    energy: float
    params: dict[str, float]
    ttl: int
    decay_rate: float = 0.02


@dataclass
class CorpsePool:
    """尸体对象池：复用 Corpse 实例，减少频繁分配。"""

    free: list[Corpse] = field(default_factory=list)
    reused_count: int = 0
    created_count: int = 0

    def acquire(
        self,
        *,
        pos: tuple[float, float],
        energy: float,
        params: dict[str, float],
        ttl: int,
        decay_rate: float = 0.02,
    ) -> Corpse:
        if self.free:
            corpse = self.free.pop()
            corpse.pos = pos
            corpse.energy = energy
            corpse.params = params
            corpse.ttl = ttl
            corpse.decay_rate = decay_rate
            self.reused_count += 1
            return corpse
        self.created_count += 1
        return Corpse(
            pos=pos,
            energy=energy,
            params=params,
            ttl=ttl,
            decay_rate=decay_rate,
        )

    def release(self, corpse: Corpse, max_pool_size: int) -> None:
        if len(self.free) >= max_pool_size:
            return
        corpse.params = {}
        corpse.energy = 0.0
        corpse.ttl = 0
        self.free.append(corpse)


@dataclass
class Network:
    neurons: dict[str, Neuron]
    queue: EventQueue = field(default_factory=EventQueue)
    replay: list[dict[str, object]] = field(default_factory=list)
    expected_state: dict[str, float] | None = None
    corpses: list[Corpse] = field(default_factory=list)
    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    energy_decay: float = 0.003  # 与 NetworkParams 默认值对齐（原值 0.005）
    max_neurons: int = 50
    max_links: int = 200
    corpse_ttl: int = 50
    scavenger_radius: float = 2.5
    input_current_scale: float = 1.5  # 与 NetworkParams 默认值对齐（原值 1.0）
    background_current: float = 0.0
    spontaneous_spike_rate: float = 0.0
    move_cost: float = 0.01
    # 长程稳定性参数
    energy_cap: float = 5.0
    adaptive_decay: bool = True
    adaptive_reproduction: bool = True
    endangered_threshold: int = 10
    # 演化参数
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "propagation": 1.0,
            "energy_efficiency": 0.5,
            "prediction_hit": 0.8,
        }
    )
    congestion_penalty: float = 0.1
    min_neurons: int = 2
    evolution_mutation_amplitude: float | None = None
    energy_budget_per_tick: float = 0.0
    # 统计跟踪
    birth_count: int = 0
    death_count: int = 0
    total_spikes: int = 0
    total_propagations: int = 0
    current_tick: int = 0
    _pe_history: list[float] = field(default_factory=list)
    _endangered_mode: bool = False
    schema_version: str = SCHEMA_VERSION
    corpse_pool: CorpsePool = field(default_factory=CorpsePool)
    corpse_pool_max_size: int = 2048

    def connect(
        self, source_id: str, target_id: str, weight: float = 1.0, delay: int = 0
    ) -> None:
        if delay < 0:
            delay = 0
        neuron = self.neurons[source_id]
        neuron.links[target_id] = {"weight": max(0.0, weight), "delay": delay}

    def connect_by_proximity(
        self, radius: float = 5.0, max_links_per_neuron: int = 5
    ) -> None:
        """基于空间距离建立连接"""
        alive = [n for n in self.neurons.values() if n.alive]
        if not alive:
            return

        # 构建空间网格加速查找
        cell_size = radius if radius > 0 else 1.0
        grid: dict[tuple[int, int], list[Neuron]] = {}
        for n in alive:
            key = (int(n.pos[0] // cell_size), int(n.pos[1] // cell_size))
            grid.setdefault(key, []).append(n)

        for source in alive:
            cx, cy = int(source.pos[0] // cell_size), int(source.pos[1] // cell_size)
            candidates: list[tuple[Neuron, float]] = []

            # 搜索相邻网格
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for target in grid.get((cx + dx, cy + dy), []):
                        if source.id == target.id:
                            continue

                        # 计算欧氏距离
                        ddx = target.pos[0] - source.pos[0]
                        ddy = target.pos[1] - source.pos[1]
                        dist = math.sqrt(ddx * ddx + ddy * ddy)

                        if dist <= radius:
                            candidates.append((target, dist))

            # 按距离排序，取最近的 N 个
            candidates.sort(key=lambda x: x[1])

            # 建立连接，权重与距离成反比，延迟与距离成正比
            links_added = 0
            for target, dist in candidates:
                if links_added >= max_links_per_neuron:
                    break
                if target.id not in source.links:
                    # 距离越近，权重越大 (0.1 ~ 1.0)
                    weight = max(0.1, 1.0 - (dist / radius))
                    # 距离越远，延迟越大 (1 ~ 5)
                    delay = max(1, int((dist / radius) * 5))
                    self.connect(source.id, target.id, weight=weight, delay=delay)
                    links_added += 1

    def add_neuron(self, neuron: Neuron) -> None:
        self.neurons[neuron.id] = neuron

    def enqueue(self, events: Iterable[Event]) -> None:
        self.queue.push_many(events)

    def tick(
        self, tick: int, dt: float = 1.0, run_evolution: bool = True
    ) -> TickSnapshot:
        self.current_tick = tick
        ready = self.queue.pop_ready(tick)
        inputs: dict[str, list[Event]] = defaultdict(list)
        for event in ready:
            if event.src_id not in self.neurons:
                continue
            inputs[event.src_id].append(event)

        emitted: list[Event] = []
        tick_spikes = 0

        # 计算自适应衰减率
        alive_count = sum(1 for n in self.neurons.values() if n.alive)
        effective_decay = self._calculate_adaptive_decay(alive_count)
        budget_share = (
            self.energy_budget_per_tick / alive_count
            if self.energy_budget_per_tick > 0 and alive_count > 0
            else 0.0
        )

        # 检查濒危模式
        if alive_count <= self.endangered_threshold and not self._endangered_mode:
            self._endangered_mode = True
        elif alive_count > self.endangered_threshold * 1.5:
            self._endangered_mode = False

        for neuron in list(self.neurons.values()):
            neuron.start_tick()
            events = inputs.get(neuron.id, [])
            input_current = self.background_current
            for event in events:
                neuron.receive(event)
                w = float(event.meta.get("weight", 1.0))
                input_current += safe_float(w * self.input_current_scale)
            spontaneous_spike = False
            if self.spontaneous_spike_rate > 0.0:
                spontaneous_spike = random.random() < self.spontaneous_spike_rate
            if spontaneous_spike:
                neuron.on_spike()
                spiked = True
            else:
                spiked = neuron.update(dt, input_current)
            if spiked:
                tick_spikes += 1
                if neuron.last_event is not None:
                    emitted.extend(neuron.emit(neuron.last_event, current_tick=tick))
                elif spontaneous_spike:
                    emitted.extend(
                        neuron.emit(
                            Event(
                                token_id=0,
                                t=tick,
                                src_id=neuron.id,
                                meta={"spontaneous": True, "weight": 1.0},
                            ),
                            current_tick=tick,
                        )
                    )
            neuron.move()

            # 应用自适应衰减（濒危模式下减半）
            decay = effective_decay * (0.5 if self._endangered_mode else 1.0)
            neuron.energy = max(0.0, neuron.energy - decay)

            if budget_share > 0.0:
                neuron.energy += budget_share

            # 应用能量上限
            if neuron.energy > self.energy_cap:
                neuron.energy = self.energy_cap

            energy_loss = max(0.0, neuron.last_energy - neuron.energy)
            reward = calculate_reward(neuron.states["novelty"], energy_loss)
            neuron.apply_reward(reward)
            neuron.finalize_tick(had_input=bool(events))

        self.queue.push_many(emitted)
        self.total_spikes += tick_spikes
        self.total_propagations += len(emitted)

        actual_state = self._aggregate_state()
        prediction_error = self._update_prediction_error(actual_state)
        self._pe_history.append(prediction_error)

        alive_neurons = [neuron for neuron in self.neurons.values() if neuron.alive]
        alive_count = len(alive_neurons)
        mean_energy = (
            sum(n.energy for n in alive_neurons) / alive_count
            if alive_count > 0
            else 0.0
        )
        snapshot = {
            "schema_version": self.schema_version,
            "tick": tick,
            "ready_events": len(ready),
            "propagation_count": len(emitted),
            "alive_neurons": alive_count,
            "total_neurons": len(self.neurons),
            "prediction_error": prediction_error,
            "tick_spikes": tick_spikes,
            "births": 0,
            "deaths": 0,
            "mean_energy": mean_energy,
        }
        self.replay.append(snapshot)

        if run_evolution:
            births, deaths = self._run_evolution_step()
            snapshot["births"] = births
            snapshot["deaths"] = deaths

        return snapshot

    def _run_evolution_step(self) -> tuple[int, int]:
        self._update_corpses()
        births = self.check_reproduction()
        deaths = self.check_death()
        self.birth_count += births
        self.death_count += deaths
        return births, deaths

    def pe_moving_average(self, window: int = 10) -> float:
        """PE 滑动均值"""
        if not self._pe_history:
            return 0.0
        recent = self._pe_history[-window:]
        return sum(recent) / len(recent)

    def active_neuron_ratio(self) -> float:
        """活跃神经元比例"""
        if not self.neurons:
            return 0.0
        alive = sum(1 for n in self.neurons.values() if n.alive)
        return alive / len(self.neurons)

    def _aggregate_state(self) -> dict[str, float]:
        alive = [neuron for neuron in self.neurons.values() if neuron.alive]
        if not alive:
            return {"energy": 0.0, "entropy": 0.0, "emotion": 0.0, "novelty": 0.0}
        size = len(alive)
        return {
            "energy": sum(neuron.states["energy"] for neuron in alive) / size,
            "entropy": sum(neuron.states["entropy"] for neuron in alive) / size,
            "emotion": sum(neuron.states["emotion"] for neuron in alive) / size,
            "novelty": sum(neuron.states["novelty"] for neuron in alive) / size,
        }

    def _update_prediction_error(self, actual_state: dict[str, float]) -> float:
        if self.expected_state is None:
            self.expected_state = dict(actual_state)
            return 0.0
        prediction_error = math.sqrt(
            sum(
                (actual_state[key] - self.expected_state.get(key, 0.0)) ** 2
                for key in actual_state
            )
        )
        prediction_error = safe_float(prediction_error)
        alpha = 0.3
        self.expected_state = {
            key: self.expected_state.get(key, 0.0)
            + alpha * (actual_state[key] - self.expected_state.get(key, 0.0))
            for key in actual_state
        }
        return prediction_error

    def check_reproduction(self) -> int:
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
            if neuron.energy < self.reproduction_threshold:
                continue
            # 拥塞惩罚: 该神经元连接过多时降低繁殖概率
            link_count = len(neuron.links)
            if link_count > 10 and random.random() < self.congestion_penalty:
                continue
            child = neuron.spawn(max(0.0, effective_mutation_scale), effective_share)
            neuron.energy = max(0.0, neuron.energy - child.energy)
            self.add_neuron(child)
            self.connect(neuron.id, child.id, weight=1.0, delay=1)
            self.connect(child.id, neuron.id, weight=1.0, delay=1)
            births += 1
        self._prune_links()
        return births

    def _calculate_adaptive_decay(self, alive_count: int) -> float:
        """根据种群密度计算自适应衰减率"""
        if not self.adaptive_decay:
            return self.energy_decay

        # 种群密度比例
        density_ratio = alive_count / self.max_neurons if self.max_neurons > 0 else 0.5

        # 密度高时增加衰减，密度低时减少衰减
        if density_ratio > 0.8:
            # 过度拥挤，增加衰减 20%
            return self.energy_decay * 1.2
        elif density_ratio < 0.3:
            # 种群稀少，减少衰减 30%
            return self.energy_decay * 0.7
        else:
            return self.energy_decay

    def _calculate_adaptive_reproduction_share(self, alive_count: int) -> float:
        """根据种群规模计算自适应繁殖成本"""
        if not self.adaptive_reproduction:
            return self.reproduction_share

        # 种群规模比例
        population_ratio = (
            alive_count / self.max_neurons if self.max_neurons > 0 else 0.5
        )

        # 接近上限时增加繁殖成本，远离上限时降低成本
        if population_ratio > 0.8:
            # 接近容量上限，增加成本（母体分享更多能量）
            return min(0.6, self.reproduction_share * 1.3)
        elif population_ratio < 0.3:
            # 种群稀少，降低成本（鼓励繁殖）
            return max(0.2, self.reproduction_share * 0.7)
        else:
            return self.reproduction_share

    def check_death(self) -> int:
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
            # 保留能量最高的死亡神经元
            dead_neurons = [
                (nid, self.neurons[nid].energy)
                for nid in dead_ids
                if nid in self.neurons
            ]
            dead_neurons.sort(key=lambda x: x[1], reverse=True)
            save_count = self.min_neurons - alive_count
            saved = {nid for nid, _ in dead_neurons[:save_count]}
            for nid in saved:
                self.neurons[nid].alive = True
                self.neurons[nid].energy = max(0.1, self.neurons[nid].energy)
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
            corpse.energy = max(0.0, corpse.energy - corpse.decay_rate)
            if corpse.energy > 0 and alive:
                cx, cy = int(corpse.pos[0] // cell_size), int(
                    corpse.pos[1] // cell_size
                )
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
            if corpse.ttl > 0 and corpse.energy > 0.0:
                updated.append(corpse)
            else:
                self.corpse_pool.release(corpse, self.corpse_pool_max_size)
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

    def save_checkpoint(self, path: str | Path, tick: int | None = None) -> Path:
        """保存网络 checkpoint（pickle）。"""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_tick = self.current_tick if tick is None else int(tick)
        payload = {
            "schema_version": self.schema_version,
            "tick": save_tick,
            "random_state": random.getstate(),
            "network": self,
        }
        with checkpoint_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return checkpoint_path

    @staticmethod
    def load_checkpoint(path: str | Path) -> "Network":
        """从 checkpoint 恢复网络，并恢复随机数状态。"""
        checkpoint_path = Path(path)
        with checkpoint_path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict):
            raise ValueError("checkpoint 格式非法：顶层必须为 dict")

        checkpoint_schema = str(payload.get("schema_version", ""))
        if checkpoint_schema:
            current_major = str(SCHEMA_VERSION).split(".", maxsplit=1)[0]
            loaded_major = checkpoint_schema.split(".", maxsplit=1)[0]
            if loaded_major != current_major:
                raise ValueError(
                    f"checkpoint schema 主版本不兼容: {checkpoint_schema} != {SCHEMA_VERSION}"
                )

        network = payload.get("network")
        if not isinstance(network, Network):
            raise ValueError("checkpoint 格式非法：缺少 network 对象")

        restored_tick = int(payload.get("tick", getattr(network, "current_tick", 0)))
        network.current_tick = max(0, restored_tick)

        random_state = payload.get("random_state")
        if random_state is not None:
            random.setstate(random_state)

        return network

    def get_summary(self) -> EngineSummary:
        """生成网络摘要（含 schema_version）"""
        alive = [n for n in self.neurons.values() if n.alive]
        return {
            "schema_version": self.schema_version,
            "total_neurons": len(self.neurons),
            "alive_neurons": len(alive),
            "total_spikes": self.total_spikes,
            "total_propagations": self.total_propagations,
            "birth_count": self.birth_count,
            "death_count": self.death_count,
            "corpse_pool_free": len(self.corpse_pool.free),
            "corpse_pool_reused": self.corpse_pool.reused_count,
            "pe_moving_avg": self.pe_moving_average(),
            "active_ratio": self.active_neuron_ratio(),
            "mean_energy": sum(n.energy for n in alive) / len(alive) if alive else 0.0,
        }
