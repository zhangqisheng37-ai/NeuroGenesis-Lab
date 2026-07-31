"""实时可视化 WebSocket 协议 (Phase 8 / P0.1)

定义 tick 数据推送格式与控制命令格式，供 `server.py`（服务端）与前端
（`web/live.html`）共享。本模块为纯 Python，不依赖 FastAPI，可独立测试；
P5 Web 管理界面复用同一套消息协议，避免重复实现。

消息方向：
- 服务端 → 客户端：``init``（初始元数据）、``tick_update``（每 tick 状态）、
  ``status``（运行状态变更）。
- 客户端 → 服务端：``pause``、``resume``、``stop``、``speed_change``。
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

from neurogenesis_lab.config import SCHEMA_VERSION

# ── 消息类型（服务端 → 客户端）──────────────────────────────
MSG_INIT = "init"
MSG_TICK = "tick_update"
MSG_STATUS = "status"

# ── 控制命令（客户端 → 服务端）──────────────────────────────
CMD_PAUSE = "pause"
CMD_RESUME = "resume"
CMD_STOP = "stop"
CMD_SPEED = "speed_change"
VALID_COMMANDS: frozenset[str] = frozenset({CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_SPEED})

# ── 运行状态 ────────────────────────────────────────────────
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_FINISHED = "finished"

# 调速边界（毫秒/ tick）
MIN_TICK_MS = 0
MAX_TICK_MS = 5000


class NeuronView(TypedDict):
    """单个神经元的可视化视图。"""

    id: str
    x: float
    y: float
    energy: float
    alive: bool
    spiked: bool


class ConnectionView(TypedDict):
    """单条连接的可视化视图。"""

    source: str
    target: str
    weight: float


def build_init_message(config: Mapping[str, Any], total_neurons: int) -> dict[str, Any]:
    """构建初始元数据消息 — 客户端连接后首先收到。

    Args:
        config: 完整仿真配置字典（``SimConfig.to_dict()``）。
        total_neurons: 初始神经元数量。

    Returns:
        ``init`` 类型消息字典。
    """
    return {
        "type": MSG_INIT,
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "total_neurons": total_neurons,
    }


def build_status_message(state: str, tick: int, tick_ms: int) -> dict[str, Any]:
    """构建运行状态消息 — 暂停/继续/结束/调速时推送。

    Args:
        state: 运行状态（``running`` / ``paused`` / ``finished``）。
        tick: 当前 tick 编号。
        tick_ms: 当前每 tick 间隔（毫秒）。

    Returns:
        ``status`` 类型消息字典。
    """
    return {
        "type": MSG_STATUS,
        "state": state,
        "tick": tick,
        "tick_ms": tick_ms,
    }


def clamp_tick_ms(value: Any) -> int:
    """将调速值裁剪到合法区间 [MIN_TICK_MS, MAX_TICK_MS]。"""
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return MIN_TICK_MS
    return max(MIN_TICK_MS, min(MAX_TICK_MS, ms))


def parse_command(raw: Any) -> tuple[str, dict[str, Any]]:
    """解析客户端控制命令，返回 (命令名, 载荷)。

    Args:
        raw: 客户端发来的原始 JSON 对象（应为 dict，含 ``type`` 字段）。

    Returns:
        (命令名, 载荷字典) 二元组。

    Raises:
        ValueError: 命令格式非法或类型未知时。
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"命令必须是对象，收到: {type(raw).__name__}")
    command = raw.get("type")
    if not isinstance(command, str) or command not in VALID_COMMANDS:
        raise ValueError(f"未知命令: {command!r}")
    payload = {key: value for key, value in raw.items() if key != "type"}
    return command, payload


class TickMessageBuilder:
    """将网络快照转换为 ``tick_update`` 消息 — 有状态，跨 tick 追踪放电。

    通过对比各神经元累计 ``spike_count`` 的前后差值判定本 tick 是否放电，
    从而在前端高亮闪烁；可选 ``max_connections`` 上限用于限制大网络的推送体积。
    """

    def __init__(self, max_connections: int = 0) -> None:
        """初始化构建器。

        Args:
            max_connections: 单次推送的最大连接数，0 表示不限。
        """
        self._prev_spike_counts: dict[str, int] = {}
        self._max_connections = max(0, max_connections)

    def build(self, tick: int, snapshot: Mapping[str, Any], network: Any) -> dict[str, Any]:
        """构建当前 tick 的推送消息。

        Args:
            tick: 当前 tick 编号。
            snapshot: ``network.tick()`` 返回的快照（含 AutoLearner 注入字段）。
            network: 网络实例，需提供 ``neurons`` 映射（id → 神经元）。

        Returns:
            ``tick_update`` 类型消息字典。
        """
        neurons: list[NeuronView] = []
        connections: list[ConnectionView] = []
        for nid, neuron in network.neurons.items():
            spike_count = int(getattr(neuron, "spike_count", 0))
            spiked = spike_count > self._prev_spike_counts.get(nid, 0)
            self._prev_spike_counts[nid] = spike_count
            pos = getattr(neuron, "pos", (0.0, 0.0))
            neurons.append(
                {
                    "id": str(nid),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "energy": float(getattr(neuron, "energy", 0.0)),
                    "alive": bool(getattr(neuron, "alive", False)),
                    "spiked": spiked,
                }
            )
            if getattr(neuron, "alive", False):
                for target_id, link in getattr(neuron, "links", {}).items():
                    if self._max_connections and len(connections) >= self._max_connections:
                        break
                    connections.append(
                        {
                            "source": str(nid),
                            "target": str(target_id),
                            "weight": float(link.get("weight", 1.0)),
                        }
                    )
        return {
            "type": MSG_TICK,
            "tick": tick,
            "neurons": neurons,
            "connections": connections,
            "stats": _extract_stats(snapshot),
        }


def _extract_stats(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """从快照提取前端统计面板所需字段（缺失字段给默认值）。"""
    return {
        "tick": snapshot.get("tick", 0),
        "alive_neurons": snapshot.get("alive_neurons", 0),
        "total_neurons": snapshot.get("total_neurons", 0),
        "tick_spikes": snapshot.get("tick_spikes", 0),
        "births": snapshot.get("births", 0),
        "deaths": snapshot.get("deaths", 0),
        "prediction_error": snapshot.get("prediction_error", 0.0),
        "mean_energy": snapshot.get("mean_energy", 0.0),
        "rule_count": snapshot.get("rule_count", 0),
        "prediction_accuracy": snapshot.get("prediction_accuracy", 0.0),
    }
