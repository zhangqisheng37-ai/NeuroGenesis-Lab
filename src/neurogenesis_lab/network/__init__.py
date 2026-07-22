"""network 子包 — 神经网络仿真引擎。

本包由原单文件 ``network.py`` 按功能拆分而来，公共 API 保持不变：

- ``engine``：``Network`` 主类（组合 dynamics / lifecycle / persistence Mixin）。
- ``event_queue``：``EventQueue`` 事件优先队列。
- ``corpse``：``Corpse`` / ``CorpsePool`` 能量回收。
- ``state``：``NetworkConfig`` / ``NetworkState`` 配置与运行时状态组件。
"""

from __future__ import annotations

from neurogenesis_lab.network.corpse import Corpse, CorpsePool
from neurogenesis_lab.network.engine import Network
from neurogenesis_lab.network.event_queue import EventQueue
from neurogenesis_lab.network.state import NetworkConfig, NetworkState

__all__ = [
    "Network",
    "EventQueue",
    "Corpse",
    "CorpsePool",
    "NetworkConfig",
    "NetworkState",
]
