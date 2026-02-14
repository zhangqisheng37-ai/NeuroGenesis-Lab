"""单元测试 — 基础日志模块"""
import io
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.logger import create_logger, log_tick


def test_create_logger_reuses_existing_handler() -> None:
    name = "neurogenesis_test_logger_reuse"

    # 清理同名 logger 的历史 handler，避免外部环境污染
    existing = logging.getLogger(name)
    for handler in list(existing.handlers):
        existing.removeHandler(handler)

    logger1 = create_logger(name)
    handler_count_1 = len(logger1.handlers)

    logger2 = create_logger(name)
    handler_count_2 = len(logger2.handlers)

    assert logger1 is logger2
    assert handler_count_1 == 1
    assert handler_count_2 == 1


def test_log_tick_writes_json_payload() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("neurogenesis_test_logger_payload")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    log_tick(logger, tick=7, module="network", message="step done", alive=3, spikes=5)

    raw = stream.getvalue().strip()
    payload = json.loads(raw)
    assert payload["tick"] == 7
    assert payload["module"] == "network"
    assert payload["message"] == "step done"
    assert payload["alive"] == 3
    assert payload["spikes"] == 5
