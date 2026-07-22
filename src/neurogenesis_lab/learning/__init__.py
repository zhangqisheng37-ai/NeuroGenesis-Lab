"""learning 子包 — 符号学习与规则发现引擎。

本包由原单文件 ``learning.py`` 按功能拆分而来，公共 API 保持不变：

- 表达式部分（``expression``）：``_parse_expression``、``MemoryBank``、
  ``PatternMiner``、``QueryEngine``。
- 规则部分（``rules``）：``RuleStore``、``RuleDiscovery``、``RuleComposer``。
- 自动学习（``auto``）：``AutoLearner``。

``mine_pattern_counts_safe`` 在此一并 re-export 以便发现；但对其打桩需针对
其实际所在模块 ``neurogenesis_lab.learning.rules``。
"""

from __future__ import annotations

from neurogenesis_lab.core_bridge import mine_pattern_counts_safe
from neurogenesis_lab.learning.auto import AutoLearner
from neurogenesis_lab.learning.expression import (
    MemoryBank,
    PatternMiner,
    QueryEngine,
    _parse_expression,
)
from neurogenesis_lab.learning.rules import (
    RuleComposer,
    RuleDiscovery,
    RuleStore,
)

__all__ = [
    "_parse_expression",
    "MemoryBank",
    "PatternMiner",
    "QueryEngine",
    "RuleStore",
    "RuleDiscovery",
    "RuleComposer",
    "AutoLearner",
    "mine_pattern_counts_safe",
]
