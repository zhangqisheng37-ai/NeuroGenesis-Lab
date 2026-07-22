"""尸体与尸体对象池 — network 子包的能量回收部分。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Corpse:
    """死亡神经元遗留的尸体，可被附近存活神经元拾取能量。

    Attributes:
        pos: 尸体空间坐标 (x, y)。
        energy: 剩余可回收能量。
        params: 死亡神经元的动力学参数快照。
        ttl: 剩余存活 tick 数，归零后移除。
        decay_rate: 每 tick 能量自然衰减量，默认 0.02。
    """

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
        """从池中获取或创建尸体实例。

        Args:
            pos: 尸体位置坐标 (x, y)。
            energy: 尸体遗留能量。
            params: 可遗传参数快照。
            ttl: 剩余存活 tick 数。
            decay_rate: 每 tick 能量衰减比率，默认 0.02。

        Returns:
            复用或新建的 Corpse 实例。
        """
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
        """将尸体实例归还对象池以供复用。

        Args:
            corpse: 待归还的尸体实例（内部状态将被重置）。
            max_pool_size: 池最大容量，超出时直接丢弃。
        """
        if len(self.free) >= max_pool_size:
            return
        corpse.params = {}
        corpse.energy = 0.0
        corpse.ttl = 0
        self.free.append(corpse)
