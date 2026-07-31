"""向量化生命周期演化 — vectorized 子包的尸体回收、自适应与繁殖/死亡部分。"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]

from neurogenesis_lab.adaptive import (
    calculate_adaptive_decay,
    calculate_adaptive_reproduction_share,
)
from neurogenesis_lab.config import (
    ENDANGERED_DECAY_DISCOUNT,
    MOVE_JITTER_RANGE,
    RESCUED_NEURON_ENERGY,
    SPAWN_LINK_DELAY,
    SPAWN_LINK_WEIGHT,
    SPAWN_POSITION_JITTER,
)
from neurogenesis_lab.evolution import (
    child_energy,
    reproduction_mask,
    select_rescued_indices,
)
from neurogenesis_lab.vectorized.state import VectorizedState


class LifecycleMixin(VectorizedState):
    """尸体状态更新、自适应参数计算与向量化繁殖/死亡。"""

    # ------------------------------------------------------------------
    # 尸体回收系统
    # ------------------------------------------------------------------

    def _update_corpses(self) -> None:
        """更新尸体状态 — TTL 递减、能量衰减、附近神经元拾取。"""
        if not np.any(self.corpse_mask):
            return

        # TTL 递减
        self.corpse_timer[self.corpse_mask] -= 1
        # 能量衰减（规则同 evolution.decay_corpse_energy 的数组化等价）
        self.corpse_energy[self.corpse_mask] = np.maximum(
            0.0, self.corpse_energy[self.corpse_mask] - self._corpse_decay_rate
        )

        # 拾荒：存活神经元吸收半径内尸体能量
        corpse_indices = np.where(self.corpse_mask & (self.corpse_energy > 0))[0]
        for ci in corpse_indices:
            cx, cy = float(self.pos[ci, 0]), float(self.pos[ci, 1])
            nearby = self.spatial.query_radius(cx, cy, self.scavenger_radius, self.pos)
            # 过滤出存活且非尸体的神经元
            alive_nearby = [idx for idx in nearby if idx != ci and self.alive[idx]]
            if alive_nearby:
                # 选择最近的一个
                dists = [
                    (self.pos[idx, 0] - cx) ** 2 + (self.pos[idx, 1] - cy) ** 2
                    for idx in alive_nearby
                ]
                nearest = alive_nearby[int(np.argmin(dists))]
                self.energy[nearest] += self.corpse_energy[ci]
                self.corpse_energy[ci] = 0.0

        # 清理过期或无能量尸体（规则同 evolution.is_corpse_expired 的数组化等价）
        expired = self.corpse_mask & ((self.corpse_timer <= 0) | (self.corpse_energy <= 0))
        self.corpse_mask[expired] = False
        self.corpse_timer[expired] = 0
        self.corpse_energy[expired] = 0.0

    # ------------------------------------------------------------------
    # 演化 tick
    # ------------------------------------------------------------------

    def _calculate_adaptive_decay(self, alive_count: int) -> float:
        """计算自适应能量衰减 — 委托 adaptive 模块（保留向后兼容 API）。

        注意：不含濒危模式折扣，折扣在 _run_vectorized_evolution 中另行叠加。

        Args:
            alive_count: 当前存活神经元数量。

        Returns:
            根据种群密度调整后的能量衰减值。
        """
        return calculate_adaptive_decay(
            self.adaptive_decay, self.energy_decay, alive_count, self.max_neurons
        )

    def _calculate_adaptive_reproduction_share(self, alive_count: int) -> float:
        """计算自适应繁殖份额 — 委托 adaptive 模块（保留向后兼容 API）。

        Args:
            alive_count: 当前存活神经元数量。

        Returns:
            根据种群规模调整后的繁殖能量分享比例。
        """
        return calculate_adaptive_reproduction_share(
            self.adaptive_reproduction,
            self.reproduction_share,
            alive_count,
            self.max_neurons,
        )

    def _run_vectorized_evolution(self, tick: int, alive_mask: np.ndarray) -> tuple[int, int]:
        _ = tick
        alive_count = int(np.sum(self.alive))

        # 移动
        move_dx = np.random.uniform(-MOVE_JITTER_RANGE, MOVE_JITTER_RANGE, self.n)
        move_dy = np.random.uniform(-MOVE_JITTER_RANGE, MOVE_JITTER_RANGE, self.n)
        self.pos[alive_mask, 0] += move_dx[alive_mask]
        self.pos[alive_mask, 1] += move_dy[alive_mask]
        self.energy[alive_mask] = np.maximum(0.0, self.energy[alive_mask] - self.move_cost)

        # 自适应衰减（含濒危折扣）
        effective_decay = self._calculate_adaptive_decay(alive_count)
        if self._endangered_mode:
            effective_decay *= ENDANGERED_DECAY_DISCOUNT
        self.energy[alive_mask] = np.maximum(0.0, self.energy[alive_mask] - effective_decay)

        # 能量上限裁切
        self.energy[alive_mask] = np.minimum(self.energy_cap, self.energy[alive_mask])

        # 死亡判定
        death_mask = alive_mask & (self.energy <= 0.0)
        alive_after = alive_count - int(np.sum(death_mask))
        if alive_after < self.min_neurons:
            death_indices = np.where(death_mask)[0]
            if len(death_indices) > 0:
                save_count = self.min_neurons - alive_after
                saved = select_rescued_indices(self.energy, death_indices, save_count)
                death_mask[saved] = False
                self.energy[saved] = RESCUED_NEURON_ENERGY

        deaths = int(np.sum(death_mask))
        # 死亡神经元 → 尸体
        dead_indices = np.where(death_mask)[0]
        for di in dead_indices:
            self.corpse_mask[di] = True
            self.corpse_timer[di] = self.corpse_ttl
            self.corpse_energy[di] = max(0.0, float(self.energy[di]))
        self.alive[death_mask] = False
        self.death_count += deaths

        # 繁殖
        births = 0
        alive_count = int(np.sum(self.alive))
        effective_share = self._calculate_adaptive_reproduction_share(alive_count)
        if alive_count < self.max_neurons:
            repro_mask = reproduction_mask(self.alive, self.energy, self.reproduction_threshold)
            repro_indices = np.where(repro_mask)[0]
            if len(repro_indices) > 0:
                new_rows: list[int] = []
                new_cols: list[int] = []
                new_weights: list[float] = []
                new_delays: list[int] = []
                for i in repro_indices:
                    if alive_count + births >= self.max_neurons:
                        break
                    dead_slots = np.where(~self.alive & ~self.corpse_mask)[0]
                    if len(dead_slots) == 0:
                        break
                    slot = int(dead_slots[0])
                    child_e = child_energy(self.energy[i], effective_share)
                    self.energy[i] -= child_e

                    self.v[slot] = self.v_rest[i]
                    self.u[slot] = 0.0
                    self.energy[slot] = child_e
                    self.alive[slot] = True
                    self.corpse_mask[slot] = False
                    self.corpse_timer[slot] = 0
                    self.corpse_energy[slot] = 0.0
                    self.pos[slot] = self.pos[i] + np.random.uniform(
                        -SPAWN_POSITION_JITTER, SPAWN_POSITION_JITTER, 2
                    )
                    self.spike_count[slot] = 0
                    self.token_memory[slot, :] = 0
                    self.token_mem_len[slot] = 0
                    self.token_mem_ptr[slot] = 0
                    self.prediction_hits[slot] = 0
                    # 继承模型类型
                    self.model_type[slot] = self.model_type[i]
                    self.tau_m[slot] = self.tau_m[i] * (
                        1.0 + np.random.uniform(-self.mutation_scale, self.mutation_scale)
                    )
                    self.v_th[slot] = self.v_th[i] * (
                        1.0 + np.random.uniform(-self.mutation_scale, self.mutation_scale)
                    )
                    new_rows.extend([int(i), slot])
                    new_cols.extend([slot, int(i)])
                    new_weights.extend([SPAWN_LINK_WEIGHT, SPAWN_LINK_WEIGHT])
                    new_delays.extend([SPAWN_LINK_DELAY, SPAWN_LINK_DELAY])
                    births += 1
                if new_rows:
                    shape = (self.n, self.n)
                    delta_w = coo_matrix((new_weights, (new_rows, new_cols)), shape=shape).tocsr()
                    delta_d = coo_matrix(
                        (new_delays, (new_rows, new_cols)), shape=shape, dtype=np.int32
                    ).tocsr()
                    self.weight_matrix = self.weight_matrix + delta_w
                    self.delay_matrix = self.delay_matrix + delta_d
        self.birth_count += births
        return births, deaths
