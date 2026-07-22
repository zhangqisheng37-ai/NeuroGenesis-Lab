"""表达式记忆与符号模式挖掘 — learning 子包的算术表达式部分。

包含：
- ``_parse_expression``：简单算术表达式解析。
- ``MemoryBank``：表达式记忆库。
- ``PatternMiner``：符号模式挖掘器。
- ``QueryEngine``：基于模式表的查询引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _parse_expression(expr: str) -> tuple[str, str, str, str] | None:
    """解析简单算术表达式字符串为四元组。

    支持格式: ``A op B = C``，其中 op 为 +、-、*、/ 之一。

    Args:
        expr: 待解析的表达式字符串，如 ``"1+2=3"``。

    Returns:
        解析成功返回 (左操作数, 运算符, 右操作数, 结果)，失败返回 None。

    Example:
        >>> _parse_expression("1+2=3")
        ('1', '+', '2', '3')
    """
    cleaned = expr.replace(" ", "")
    if "=" not in cleaned:
        return None
    left, right = cleaned.split("=", 1)
    if len(left) < 3:
        return None
    for index, char in enumerate(left):
        if char in {"+", "-", "*", "/"}:
            a = left[:index]
            op = char
            b = left[index + 1 :]
            if not a or not b:
                return None
            return a, op, b, right
    return None


@dataclass
class MemoryBank:
    """表达式记忆库 — 存储已观察到的算术表达式字符串。

    Attributes:
        expressions: 已存储的表达式列表。
    """

    expressions: list[str] = field(default_factory=list)

    def add(self, expr: str) -> None:
        """添加一条表达式。

        Args:
            expr: 表达式字符串。
        """
        self.expressions.append(expr)

    def add_many(self, exprs: list[str]) -> None:
        """批量添加表达式。

        Args:
            exprs: 表达式字符串列表。
        """
        for expr in exprs:
            self.add(expr)


@dataclass
class PatternMiner:
    """符号模式挖掘器 — 从表达式中提取 (操作数, 运算符, 操作数) -> 结果 的映射。

    Attributes:
        table: 模式查找表，键为 (a, op, b) 三元组，值为结果字符串。
        support_count: 各模式的出现次数统计。
    """

    table: dict[tuple[str, str, str], str] = field(default_factory=dict)
    # 置信度/支持度统计
    support_count: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def fit(self, expressions: list[str]) -> None:
        """从表达式列表中提取模式并更新查找表。

        Args:
            expressions: 表达式字符串列表，如 ``["1+2=3", "3*4=12"]``。
        """
        for expr in expressions:
            parsed = _parse_expression(expr)
            if parsed is None:
                continue
            left_a, op, left_b, right = parsed
            if right:
                key = (left_a, op, left_b)
                self.table[key] = right
                self.support_count[key] = self.support_count.get(key, 0) + 1

    def lookup(self, a: str, op: str, b: str) -> str | None:
        """查找模式表中的结果。

        Args:
            a: 左操作数。
            op: 运算符。
            b: 右操作数。

        Returns:
            匹配的结果字符串，未找到返回 None。
        """
        return self.table.get((a, op, b))

    def confidence(self, a: str, op: str, b: str) -> float:
        """返回指定模式的置信度。

        Args:
            a: 左操作数。
            op: 运算符。
            b: 右操作数。

        Returns:
            置信度值，范围 [0, 1]，无匹配时返回 0.0。
        """
        key = (a, op, b)
        count = self.support_count.get(key, 0)
        if count == 0:
            return 0.0
        total = sum(self.support_count.values())
        return count / total if total > 0 else 0.0


@dataclass
class QueryEngine:
    """查询引擎 — 基于 PatternMiner 回答算术表达式查询。

    Attributes:
        miner: 底层模式挖掘器实例。
    """

    miner: PatternMiner

    def query(self, expr: str) -> str:
        """查询表达式结果。

        Args:
            expr: 查询表达式，右侧为 ``?`` 或空，如 ``"1+2=?"``。

        Returns:
            匹配的结果字符串，未找到返回 ``"unknown"``。
        """
        parsed = _parse_expression(expr)
        if parsed is None:
            return "unknown"
        left_a, op, left_b, right = parsed
        if right not in {"?", ""}:
            return "unknown"
        result = self.miner.lookup(left_a, op, left_b)
        return result if result is not None else "unknown"

    def query_with_explanation(self, expr: str) -> dict:
        """查询并返回解释信息，包含置信度和命中规则。

        Args:
            expr: 查询表达式，右侧为 ``?`` 或空。

        Returns:
            包含 answer、reason、confidence、rules_hit 的字典。
        """
        parsed = _parse_expression(expr)
        if parsed is None:
            return {
                "answer": "unknown",
                "reason": "无法解析表达式",
                "confidence": 0.0,
                "rules_hit": [],
            }
        left_a, op, left_b, right = parsed
        if right not in {"?", ""}:
            return {
                "answer": "unknown",
                "reason": "右侧非占位符",
                "confidence": 0.0,
                "rules_hit": [],
            }
        result = self.miner.lookup(left_a, op, left_b)
        if result is not None:
            conf = self.miner.confidence(left_a, op, left_b)
            return {
                "answer": result,
                "reason": f"查表命中: ({left_a}, {op}, {left_b}) -> {result}",
                "confidence": conf,
                "rules_hit": [
                    {
                        "type": "pattern",
                        "key": f"{left_a}{op}{left_b}",
                        "result": result,
                    }
                ],
            }
        return {
            "answer": "unknown",
            "reason": "未找到匹配模式",
            "confidence": 0.0,
            "rules_hit": [],
        }
