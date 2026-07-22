"""向量化网络 checkpoint 持久化 — vectorized 子包的序列化部分。"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.vectorized.state import VectorizedState

if TYPE_CHECKING:
    from neurogenesis_lab.vectorized.engine import VectorizedNetwork


class PersistenceMixin(VectorizedState):
    """向量化网络完整状态的保存与恢复。"""

    def save_checkpoint(self, path: str | Path, save_tick: int = 0) -> None:
        """将向量化网络完整状态序列化为 pickle 文件。

        Args:
            path: checkpoint 文件路径。
            save_tick: 当前 tick 编号，记入 payload。
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "tick": save_tick,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "network": self,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> tuple["VectorizedNetwork", int]:
        """从 pickle checkpoint 恢复向量化网络实例。

        Args:
            path: checkpoint 文件路径。

        Returns:
            (network, tick) 恢复的网络实例和保存时的 tick 编号。

        Raises:
            ValueError: schema 版本不兼容时抛出。
            TypeError: 序列化对象类型不匹配时抛出。
        """
        filepath = Path(path)
        with open(filepath, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        saved_version = payload.get("schema_version", "0.0")
        current_major = SCHEMA_VERSION.split(".")[0]
        saved_major = saved_version.split(".")[0]
        if current_major != saved_major:
            raise ValueError(
                f"Schema 版本不兼容：当前 {SCHEMA_VERSION}，checkpoint {saved_version}"
            )
        net = payload.get("network")
        if not isinstance(net, cls):
            raise TypeError(
                f"Checkpoint 中的对象类型不匹配：期望 {cls.__name__}，实际 {type(net).__name__}"
            )
        tick = payload.get("tick", 0)
        random.setstate(payload["random_state"])
        if "numpy_random_state" in payload:
            np.random.set_state(payload["numpy_random_state"])
        return cast("VectorizedNetwork", net), tick
