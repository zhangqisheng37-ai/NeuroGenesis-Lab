"""N-gram 规则存储、发现与组合推理 — learning 子包的规则部分。

包含：
- ``RuleStore``：LRU 淘汰的 trigram/bigram 规则存储。
- ``RuleDiscovery``：从 token 序列挖掘 N-gram 规则（委托 core_bridge）。
- ``RuleComposer``：基于规则库的序列填充推理。

注意：``mine_pattern_counts_safe`` 在本模块命名空间导入，因此对其打桩应
使用 ``neurogenesis_lab.learning.rules.mine_pattern_counts_safe``。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from neurogenesis_lab.core_bridge import mine_pattern_counts_safe


@dataclass
class RuleStore:
    """规则存储 — 使用 OrderedDict 实现 LRU 淘汰策略

    trigram/bigram 规则按访问顺序排列，最近访问的排在末尾。
    当规则数超过容量上限时，淘汰最久未访问的规则。
    """

    trigram_support: OrderedDict[tuple[int, int, int], int] = field(default_factory=OrderedDict)
    bigram_support: OrderedDict[tuple[int, int], int] = field(default_factory=OrderedDict)
    # 置信度统计
    trigram_total: int = 0
    bigram_total: int = 0

    def __post_init__(self) -> None:
        # 确保传入的普通 dict 被转换为 OrderedDict
        if not isinstance(self.trigram_support, OrderedDict):
            self.trigram_support = OrderedDict(self.trigram_support)
        if not isinstance(self.bigram_support, OrderedDict):
            self.bigram_support = OrderedDict(self.bigram_support)
        self.trigram_total = sum(self.trigram_support.values())
        self.bigram_total = sum(self.bigram_support.values())

    def _touch_trigram(self, key: tuple[int, int, int]) -> None:
        """将 trigram 规则移到末尾（标记为最近访问）。"""
        if key in self.trigram_support:
            self.trigram_support.move_to_end(key)

    def _touch_bigram(self, key: tuple[int, int]) -> None:
        """将 bigram 规则移到末尾（标记为最近访问）。"""
        if key in self.bigram_support:
            self.bigram_support.move_to_end(key)

    def trigram_confidence(self, trigram: tuple[int, int, int]) -> float:
        """返回 trigram 规则的置信度。

        Args:
            trigram: (左, 中, 右) token 三元组。

        Returns:
            置信度值，范围 [0, 1]。
        """
        if self.trigram_total == 0:
            return 0.0
        self._touch_trigram(trigram)
        return self.trigram_support.get(trigram, 0) / self.trigram_total

    def bigram_confidence(self, bigram: tuple[int, int]) -> float:
        """返回 bigram 规则的置信度。

        Args:
            bigram: (左, 右) token 二元组。

        Returns:
            置信度值，范围 [0, 1]。
        """
        if self.bigram_total == 0:
            return 0.0
        self._touch_bigram(bigram)
        return self.bigram_support.get(bigram, 0) / self.bigram_total

    def most_common_middle(self, left: int, right: int) -> int | None:
        """查找给定左右 token 之间最常见的中间 token。

        Args:
            left: 左侧 token ID。
            right: 右侧 token ID。

        Returns:
            支持度最高的中间 token ID，无匹配返回 None。
        """
        candidates = [
            (mid, support)
            for (a, mid, b), support in self.trigram_support.items()
            if a == left and b == right
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[1])
        self._touch_trigram((left, best[0], right))
        return best[0]

    def most_common_next(self, left: int) -> int | None:
        """查找给定 token 之后最常见的后继 token。

        Args:
            left: 前驱 token ID。

        Returns:
            支持度最高的后继 token ID，无匹配返回 None。
        """
        candidates = [(b, support) for (a, b), support in self.bigram_support.items() if a == left]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[1])
        self._touch_bigram((left, best[0]))
        return best[0]

    def top_candidates_middle(self, left: int, right: int, k: int = 3) -> list[tuple[int, int]]:
        """返回前 k 个候选中间 token。

        Args:
            left: 左侧 token ID。
            right: 右侧 token ID。
            k: 返回候选数量，默认 3。

        Returns:
            按支持度降序排列的 (token, support) 列表。
        """
        candidates = [
            (mid, support)
            for (a, mid, b), support in self.trigram_support.items()
            if a == left and b == right
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    def top_candidates_next(self, left: int, k: int = 3) -> list[tuple[int, int]]:
        """返回前 k 个候选后继 token。

        Args:
            left: 前驱 token ID。
            k: 返回候选数量，默认 3。

        Returns:
            按支持度降序排列的 (token, support) 列表。
        """
        candidates = [(b, support) for (a, b), support in self.bigram_support.items() if a == left]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]


@dataclass
class RuleDiscovery:
    """N-gram 规则发现器 — 从 token 序列样本中挖掘 bigram/trigram 模式。

    Attributes:
        min_support: 最小支持度阈值，低于此值的模式被过滤，默认 2。
    """

    min_support: int = 2

    def discover(self, samples: list[list[int]]) -> RuleStore:
        """从样本序列中发现规则并返回规则库。

        Args:
            samples: token 序列列表，每个元素为一条 token ID 序列。

        Returns:
            包含 trigram 和 bigram 规则的 RuleStore 实例。
        """
        filtered_trigrams, filtered_bigrams, _backend = mine_pattern_counts_safe(
            samples, self.min_support
        )
        return RuleStore(
            trigram_support=OrderedDict(filtered_trigrams),
            bigram_support=OrderedDict(filtered_bigrams),
        )


@dataclass
class RuleComposer:
    """规则组合推理器 — 基于规则库填充序列中的缺失 token。

    支持直接算术运算（加减乘）和基于 trigram/bigram 规则的统计推理。

    Attributes:
        store: 底层规则库。
        conflict_strategy: 冲突处理策略，``"highest_support"`` 或 ``"weighted_score"``。
    """

    store: RuleStore
    # 冲突处理策略: "highest_support" | "weighted_score"
    conflict_strategy: str = "highest_support"

    _ASCII_DIGIT_0 = ord("0")  # 48
    _ASCII_DIGIT_9 = ord("9")  # 57
    _ASCII_EQUALS = ord("=")  # 61

    @staticmethod
    def _digit_value(token: int) -> int | None:
        if RuleComposer._ASCII_DIGIT_0 <= token <= RuleComposer._ASCII_DIGIT_9:
            return token - RuleComposer._ASCII_DIGIT_0
        return None

    def _apply_arithmetic(
        self,
        sequence: list[int],
        missing_token: int,
        op: str,
        op_char: int,
        calc: Callable[[int, int], int],
    ) -> list[int] | None:
        """通用算术运算填充方法"""
        missing_indices = [index for index, token in enumerate(sequence) if token == missing_token]
        if missing_indices != [len(sequence) - 1]:
            return None
        if len(sequence) < 5:
            return None
        if sequence[1] != op_char or sequence[3] != RuleComposer._ASCII_EQUALS:
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

    def _apply_simple_subtraction(
        self, sequence: list[int], missing_token: int
    ) -> list[int] | None:
        return self._apply_arithmetic(sequence, missing_token, "-", ord("-"), lambda a, b: a - b)

    def _apply_simple_multiply(self, sequence: list[int], missing_token: int) -> list[int] | None:
        return self._apply_arithmetic(sequence, missing_token, "*", ord("*"), lambda a, b: a * b)

    def apply(self, sequence: list[int], missing_token: int = -1) -> list[int]:
        """填充序列中的缺失 token。

        优先尝试直接算术运算，失败后回退到 trigram/bigram 规则推理。

        Args:
            sequence: 输入 token 序列，缺失位置用 missing_token 标记。
            missing_token: 缺失占位符值，默认 -1。

        Returns:
            填充后的 token 序列副本。
        """
        # 尝试直接运算
        for op_handler in [
            self._apply_simple_addition,
            self._apply_simple_subtraction,
            self._apply_simple_multiply,
        ]:
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
        """推理并返回每步命中的规则详情。

        Args:
            sequence: 输入 token 序列。
            missing_token: 缺失占位符值，默认 -1。

        Returns:
            包含 filled（填充结果）和 rules_hit（命中规则列表）的字典。
        """
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
                    rules_hit.append(
                        {
                            "type": "trigram",
                            "pos": index,
                            "left": left,
                            "right": right,
                            "filled": candidate,
                            "support": self.store.trigram_support.get((left, candidate, right), 0),
                        }
                    )
                    filled[index] = candidate
                    continue
            if left is not None:
                candidate = self.store.most_common_next(left)
                if candidate is not None:
                    rules_hit.append(
                        {
                            "type": "bigram",
                            "pos": index,
                            "left": left,
                            "filled": candidate,
                            "support": self.store.bigram_support.get((left, candidate), 0),
                        }
                    )
                    filled[index] = candidate

        return {"filled": filled, "rules_hit": rules_hit}
