from __future__ import annotations

from dataclasses import dataclass, field


def _parse_expression(expr: str) -> tuple[str, str, str, str] | None:
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
    expressions: list[str] = field(default_factory=list)

    def add(self, expr: str) -> None:
        self.expressions.append(expr)

    def add_many(self, exprs: list[str]) -> None:
        for expr in exprs:
            self.add(expr)


@dataclass
class PatternMiner:
    table: dict[tuple[str, str, str], str] = field(default_factory=dict)

    def fit(self, expressions: list[str]) -> None:
        for expr in expressions:
            parsed = _parse_expression(expr)
            if parsed is None:
                continue
            left_a, op, left_b, right = parsed
            if right:
                self.table[(left_a, op, left_b)] = right

    def lookup(self, a: str, op: str, b: str) -> str | None:
        return self.table.get((a, op, b))


@dataclass
class QueryEngine:
    miner: PatternMiner

    def query(self, expr: str) -> str:
        parsed = _parse_expression(expr)
        if parsed is None:
            return "unknown"
        left_a, op, left_b, right = parsed
        if right not in {"?", ""}:
            return "unknown"
        result = self.miner.lookup(left_a, op, left_b)
        return result if result is not None else "unknown"
