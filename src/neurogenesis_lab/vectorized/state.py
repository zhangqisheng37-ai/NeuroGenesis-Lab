"""向量化网络状态 — vectorized 子包的字段与基础工具部分。

``VectorizedState`` 承载 ``VectorizedNetwork`` 的全部 dataclass 字段
（状态向量、参数向量、稀疏连接矩阵、配置、统计等），并提供仅操作自身状态的
小工具方法。各功能 Mixin 继承本类以获得字段类型信息；``VectorizedNetwork``
经多重继承从本类获得完整字段集合，无需重复声明。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.vectorized.spatial import SpatialGrid


@dataclass
class VectorizedState:
    """向量化网络的全部字段容器 + 基础状态工具方法。

    Attributes:
        n: 神经元总数（含已死亡槽位）。
        v: 膜电位向量，形状 (n,)，单位 mV。
        u: 恢复变量向量，形状 (n,)。
        energy: 能量向量，形状 (n,)。
        alive: 存活掩码向量，形状 (n,)，dtype bool。
        pos: 位置矩阵，形状 (n, 2)。
        spike_count: 累计放电次数向量，形状 (n,)。
        weight_matrix: 连接权重稀疏矩阵 (n, n)。
        delay_matrix: 传播延迟稀疏矩阵 (n, n)。
        energy_decay: 每 tick 基础能量衰减量，默认 0.005。
        schema_version: 数据格式版本号。
    """

    # 神经元状态向量
    n: int = 0
    v: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    u: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alive: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spike_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    token_memory: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    token_mem_len: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    token_mem_ptr: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    prediction_hits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # 参数向量
    v_rest: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    v_reset: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    v_th: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    tau_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    r_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spike_cost: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # 连接矩阵 (scipy 稀疏)
    weight_matrix: csr_matrix | None = None
    delay_matrix: csr_matrix | None = None

    # 配置
    energy_decay: float = 0.005
    move_cost: float = 0.01
    input_current_scale: float = 1.0
    max_neurons: int = 1000
    scavenger_radius: float = 2.5
    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    min_neurons: int = 2
    energy_cap: float = 5.0
    corpse_ttl: int = 50
    adaptive_decay: bool = True
    adaptive_reproduction: bool = True
    endangered_threshold: int = 10

    # 多模型类型向量（0=LIF, 1=Izhikevich, 2=AdEx）
    model_type: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    # Izhikevich 额外参数
    izh_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    izh_b: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    izh_c: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    izh_d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # AdEx 额外参数
    adex_delta_t: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    adex_tau_w: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    adex_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    adex_b_param: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    adex_v_peak: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # 适应电流 w（AdEx/Izh 恢复变量统一使用 self.u）

    # 尸体回收系统
    corpse_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    corpse_timer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    corpse_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    _corpse_decay_rate: float = 0.02

    # 濒危模式标志
    _endangered_mode: bool = False

    # 空间索引
    spatial: SpatialGrid = field(default_factory=SpatialGrid)
    # 事件调度（tick -> 输入电流向量）
    scheduled_inputs: dict[int, np.ndarray] = field(default_factory=dict)
    _last_spike_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))

    # 统计
    total_spikes: int = 0
    total_propagations: int = 0
    total_ticks: int = 0
    birth_count: int = 0
    death_count: int = 0
    _pe_history: list[float] = field(default_factory=list)
    expected_state: np.ndarray | None = None
    schema_version: str = SCHEMA_VERSION

    # ── 基础状态工具 ─────────────────────────────────────
    def _resolve_neuron_index(self, src_id: int | str) -> int | None:
        if isinstance(src_id, int):
            idx = src_id
        elif isinstance(src_id, str):
            text = src_id.strip()
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                idx = int(text)
            else:
                return None
        else:
            return None
        if 0 <= idx < self.n:
            return idx
        return None

    def _schedule_currents(self, tick: int, currents: np.ndarray) -> None:
        if tick in self.scheduled_inputs:
            self.scheduled_inputs[tick] += currents
        else:
            self.scheduled_inputs[tick] = currents.copy()

    def _write_token(self, idx: int, token_id: int) -> None:
        if self.n <= 0 or idx < 0 or idx >= self.n:
            return
        if self.token_memory.size == 0:
            return
        mem_size = self.token_memory.shape[1]
        if mem_size <= 0:
            return
        ptr = int(self.token_mem_ptr[idx] % mem_size)
        self.token_memory[idx, ptr] = int(token_id)
        self.token_mem_ptr[idx] = (ptr + 1) % mem_size
        if self.token_mem_len[idx] < mem_size:
            self.token_mem_len[idx] += 1

    def _cleanup_scheduled_inputs(self, current_tick: int) -> None:
        stale_ticks = [k for k in self.scheduled_inputs if k < current_tick]
        for stale_tick in stale_ticks:
            self.scheduled_inputs.pop(stale_tick, None)
