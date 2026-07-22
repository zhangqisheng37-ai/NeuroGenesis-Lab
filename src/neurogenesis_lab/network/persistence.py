"""网络 checkpoint 持久化 — network 子包的序列化部分。"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.network._base import _NetworkAttrs

if TYPE_CHECKING:
    from neurogenesis_lab.network.engine import Network


class NetworkPersistenceMixin(_NetworkAttrs):
    """checkpoint 的保存与恢复。"""

    def save_checkpoint(self, path: str | Path, tick: int | None = None) -> Path:
        """保存网络 checkpoint（pickle 格式）。

        Args:
            path: 输出文件路径。
            tick: 保存的 tick 编号，None 时使用 current_tick。

        Returns:
            实际写入的文件路径。

        Raises:
            OSError: 文件写入失败时抛出。
        """
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
        """从 checkpoint 恢复网络，并恢复随机数状态。

        Args:
            path: checkpoint 文件路径。

        Returns:
            恢复后的 Network 实例。

        Raises:
            ValueError: checkpoint 格式非法或 schema 主版本不兼容时抛出。
        """
        # 惰性导入避免子模块循环依赖（engine 导入本模块）
        from neurogenesis_lab.network.engine import Network

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
