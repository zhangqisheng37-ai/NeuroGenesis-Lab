"""事件优先队列 — network 子包的事件调度部分。"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

from neurogenesis_lab.encoding import Event


@dataclass
class EventQueue:
    """基于最小堆的事件优先队列，按时间戳排序。

    Attributes:
        _heap: 内部堆存储，元素为 (时间戳, 序号, 事件) 三元组。
        _seq: 单调递增序号，保证同时间戳事件的 FIFO 顺序。
    """

    _heap: list[tuple[int, int, Event]] = field(default_factory=list)
    _seq: int = 0

    def push(self, event: Event) -> None:
        """将事件压入队列。

        Args:
            event: 待入队的事件，按 event.t 排序。
        """
        heapq.heappush(self._heap, (event.t, self._seq, event))
        self._seq += 1

    def push_many(self, events: Iterable[Event]) -> None:
        """批量压入事件。

        Args:
            events: 事件可迭代对象。
        """
        for event in events:
            self.push(event)

    def pop_ready(self, tick: int) -> list[Event]:
        """弹出所有时间戳 <= tick 的就绪事件。

        Args:
            tick: 当前时间步。

        Returns:
            按时间戳升序排列的就绪事件列表。
        """
        ready: list[Event] = []
        while self._heap and self._heap[0][0] <= tick:
            _, _, event = heapq.heappop(self._heap)
            ready.append(event)
        return ready

    def __len__(self) -> int:
        return len(self._heap)
