import json
import logging
from typing import Any


def create_logger(name: str = "neurogenesis") -> logging.Logger:
    """创建或获取命名日志器，自动配置 StreamHandler。

    若同名日志器已有 handler 则直接返回，避免重复添加。

    Args:
        name: 日志器名称，默认 ``"neurogenesis"``。

    Returns:
        配置好的 Logger 实例。

    Example:
        >>> logger = create_logger("my_experiment")
        >>> logger.info("启动仿真")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_tick(
    logger: logging.Logger,
    tick: int,
    module: str,
    message: str,
    **extra: Any,
) -> None:
    """以 JSON 格式记录一条 tick 级日志。

    Args:
        logger: 目标日志器。
        tick: 当前 tick 编号。
        module: 模块名称标识。
        message: 日志消息文本。
        **extra: 附加键值对，合并到 JSON 输出中。
    """
    payload = {"tick": tick, "module": module, "message": message, **extra}
    logger.info(json.dumps(payload, ensure_ascii=False))
