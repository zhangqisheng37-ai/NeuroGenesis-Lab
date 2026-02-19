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
    # 置信度/支持度统计
    support_count: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def fit(self, expressions: list[str]) -> None:
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
        return self.table.get((a, op, b))

    def confidence(self, a: str, op: str, b: str) -> float:
        """返回查询的置信度 (0~1)"""
        key = (a, op, b)
        count = self.support_count.get(key, 0)
        if count == 0:
            return 0.0
        total = sum(self.support_count.values())
        return count / total if total > 0 else 0.0


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

    def query_with_explanation(self, expr: str) -> dict:
        """查询并返回解释信息"""
        parsed = _parse_expression(expr)
        if parsed is None:
            return {"answer": "unknown", "reason": "无法解析表达式", "confidence": 0.0, "rules_hit": []}
        left_a, op, left_b, right = parsed
        if right not in {"?", ""}:
            return {"answer": "unknown", "reason": "右侧非占位符", "confidence": 0.0, "rules_hit": []}
        result = self.miner.lookup(left_a, op, left_b)
        if result is not None:
            conf = self.miner.confidence(left_a, op, left_b)
            return {
                "answer": result,
                "reason": f"查表命中: ({left_a}, {op}, {left_b}) -> {result}",
                "confidence": conf,
                "rules_hit": [{"type": "pattern", "key": f"{left_a}{op}{left_b}", "result": result}],
            }
        return {"answer": "unknown", "reason": "未找到匹配模式", "confidence": 0.0, "rules_hit": []}


@dataclass
class RuleStore:
    trigram_support: dict[tuple[int, int, int], int] = field(default_factory=dict)
    bigram_support: dict[tuple[int, int], int] = field(default_factory=dict)
    # 置信度统计
    trigram_total: int = 0
    bigram_total: int = 0

    def __post_init__(self) -> None:
        self.trigram_total = sum(self.trigram_support.values())
        self.bigram_total = sum(self.bigram_support.values())

    def trigram_confidence(self, trigram: tuple[int, int, int]) -> float:
        if self.trigram_total == 0:
            return 0.0
        return self.trigram_support.get(trigram, 0) / self.trigram_total

    def bigram_confidence(self, bigram: tuple[int, int]) -> float:
        if self.bigram_total == 0:
            return 0.0
        return self.bigram_support.get(bigram, 0) / self.bigram_total

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

    def top_candidates_middle(self, left: int, right: int, k: int = 3) -> list[tuple[int, int]]:
        """返回前 k 个候选中间 token (token, support)"""
        candidates = [
            (mid, support)
            for (a, mid, b), support in self.trigram_support.items()
            if a == left and b == right
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    def top_candidates_next(self, left: int, k: int = 3) -> list[tuple[int, int]]:
        """返回前 k 个候选后继 token"""
        candidates = [
            (b, support) for (a, b), support in self.bigram_support.items() if a == left
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]


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
    # 冲突处理策略: "highest_support" | "weighted_score"
    conflict_strategy: str = "highest_support"

    @staticmethod
    def _digit_value(token: int) -> int | None:
        if 48 <= token <= 57:
            return token - 48
        return None

    def _apply_arithmetic(self, sequence: list[int], missing_token: int, op: str, op_char: int, calc: callable) -> list[int] | None:
        """通用算术运算填充方法"""
        missing_indices = [index for index, token in enumerate(sequence) if token == missing_token]
        if missing_indices != [len(sequence) - 1]:
            return None
        if len(sequence) < 5:
            return None
        if sequence[1] != op_char or sequence[3] != ord("="):
            return None
        left = self._digit_value(sequence[0])
        right = self._digit_value(sequence[2])
        if left is None or right is None:
            return None
        result = calc(left, right)
        if result < 0 or result >= 10:
            return None
        filled = list(sequence)
        filled[-1] = ord(str(result))
        return filled

    def _apply_simple_addition(self, sequence: list[int], missing_token: int) -> list[int] | None:
        return self._apply_arithmetic(sequence, missing_token, "+", ord("+"), lambda a, b: a + b)

    def _apply_simple_subtraction(self, sequence: list[int], missing_token: int) -> list[int] | None:
        return self._apply_arithmetic(sequence, missing_token, "-", ord("-"), lambda a, b: a - b)

    def _apply_simple_multiply(self, sequence: list[int], missing_token: int) -> list[int] | None:
        return self._apply_arithmetic(sequence, missing_token, "*", ord("*"), lambda a, b: a * b)

    def apply(self, sequence: list[int], missing_token: int = -1) -> list[int]:
        # 尝试直接运算
        for op_handler in [self._apply_simple_addition, self._apply_simple_subtraction, self._apply_simple_multiply]:
            direct = op_handler(sequence, missing_token)
            if direct is not None:
                return direct

        # 基于规则填充
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

    def apply_with_explanation(self, sequence: list[int], missing_token: int = -1) -> dict:
        """推理并返回每步命中的规则"""
        rules_hit: list[dict] = []

        for op_name, op_handler in [
            ("addition", self._apply_simple_addition),
            ("subtraction", self._apply_simple_subtraction),
            ("multiply", self._apply_simple_multiply),
        ]:
            direct = op_handler(sequence, missing_token)
            if direct is not None:
                rules_hit.append({"type": "arithmetic", "operation": op_name})
                return {"filled": direct, "rules_hit": rules_hit}

        filled = list(sequence)
        for index, token in enumerate(filled):
            if token != missing_token:
                continue
            left = filled[index - 1] if index - 1 >= 0 else None
            right = filled[index + 1] if index + 1 < len(filled) else None

            if left is not None and right is not None and right != missing_token:
                candidate = self.store.most_common_middle(left, right)
                if candidate is not None:
                    rules_hit.append({
                        "type": "trigram", "pos": index,
                        "left": left, "right": right, "filled": candidate,
                        "support": self.store.trigram_support.get((left, candidate, right), 0),
                    })
                    filled[index] = candidate
                    continue
            if left is not None:
                candidate = self.store.most_common_next(left)
                if candidate is not None:
                    rules_hit.append({
                        "type": "bigram", "pos": index,
                        "left": left, "filled": candidate,
                        "support": self.store.bigram_support.get((left, candidate), 0),
                    })
                    filled[index] = candidate

        return {"filled": filled, "rules_hit": rules_hit}
