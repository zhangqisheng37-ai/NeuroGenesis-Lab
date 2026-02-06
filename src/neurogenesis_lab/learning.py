from __future__ import annotations

from collections import Counter
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


@dataclass
class RuleStore:
    trigram_support: dict[tuple[int, int, int], int] = field(default_factory=dict)
    bigram_support: dict[tuple[int, int], int] = field(default_factory=dict)

    def most_common_middle(self, left: int, right: int) -> int | None:
        candidates = [
            (mid, support)
            for (a, mid, b), support in self.trigram_support.items()
            if a == left and b == right
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    def most_common_next(self, left: int) -> int | None:
        candidates = [
            (b, support) for (a, b), support in self.bigram_support.items() if a == left
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]


@dataclass
class RuleDiscovery:
    min_support: int = 2

    def discover(self, samples: list[list[int]]) -> RuleStore:
        trigram_support: Counter[tuple[int, int, int]] = Counter()
        bigram_support: Counter[tuple[int, int]] = Counter()

        for sequence in samples:
            trigrams = set()
            bigrams = set()
            for index in range(len(sequence) - 2):
                trigrams.add((sequence[index], sequence[index + 1], sequence[index + 2]))
            for index in range(len(sequence) - 1):
                bigrams.add((sequence[index], sequence[index + 1]))
            for trigram in trigrams:
                trigram_support[trigram] += 1
            for bigram in bigrams:
                bigram_support[bigram] += 1

        filtered_trigrams = {
            trigram: support
            for trigram, support in trigram_support.items()
            if support >= self.min_support
        }
        filtered_bigrams = {
            bigram: support
            for bigram, support in bigram_support.items()
            if support >= self.min_support
        }
        return RuleStore(trigram_support=filtered_trigrams, bigram_support=filtered_bigrams)


@dataclass
class RuleComposer:
    store: RuleStore

    @staticmethod
    def _digit_value(token: int) -> int | None:
        if 48 <= token <= 57:
            return token - 48
        return None

    def _apply_simple_addition(self, sequence: list[int], missing_token: int) -> list[int] | None:
        missing_indices = [index for index, token in enumerate(sequence) if token == missing_token]
        if missing_indices != [len(sequence) - 1]:
            return None
        if len(sequence) < 5:
            return None
        if sequence[1] != ord("+") or sequence[3] != ord("="):
            return None
        left = self._digit_value(sequence[0])
        right = self._digit_value(sequence[2])
        if left is None or right is None:
            return None
        result = left + right
        if result >= 10:
            return None
        filled = list(sequence)
        filled[-1] = ord(str(result))
        return filled

    def apply(self, sequence: list[int], missing_token: int = -1) -> list[int]:
        direct = self._apply_simple_addition(sequence, missing_token)
        if direct is not None:
            return direct
        filled = list(sequence)
        for index, token in enumerate(filled):
            if token != missing_token:
                continue
            left = filled[index - 1] if index - 1 >= 0 else None
            right = filled[index + 1] if index + 1 < len(filled) else None
            if left is not None and right is not None and right != missing_token:
                candidate = self.store.most_common_middle(left, right)
                if candidate is not None:
                    filled[index] = candidate
                    continue
            if left is not None:
                candidate = self.store.most_common_next(left)
                if candidate is not None:
                    filled[index] = candidate
        return filled
