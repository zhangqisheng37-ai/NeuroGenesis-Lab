from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from neurogenesis_lab.config import ASCII_PRINTABLE_MIN, ASCII_PRINTABLE_MAX
from neurogenesis_lab.core_bridge import mine_pattern_counts_safe
from neurogenesis_lab.protocols import TickSnapshot

if TYPE_CHECKING:
    from neurogenesis_lab.network import Network


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


@dataclass
class RuleStore:
    """规则存储 — 使用 OrderedDict 实现 LRU 淘汰策略

    trigram/bigram 规则按访问顺序排列，最近访问的排在末尾。
    当规则数超过容量上限时，淘汰最久未访问的规则。
    """

    trigram_support: OrderedDict[tuple[int, int, int], int] = field(
        default_factory=OrderedDict
    )
    bigram_support: OrderedDict[tuple[int, int], int] = field(
        default_factory=OrderedDict
    )
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
        candidates = [
            (b, support) for (a, b), support in self.bigram_support.items() if a == left
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[1])
        self._touch_bigram((left, best[0]))
        return best[0]

    def top_candidates_middle(
        self, left: int, right: int, k: int = 3
    ) -> list[tuple[int, int]]:
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
        candidates = [
            (b, support) for (a, b), support in self.bigram_support.items() if a == left
        ]
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
        missing_indices = [
            index for index, token in enumerate(sequence) if token == missing_token
        ]
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

    def _apply_simple_addition(
        self, sequence: list[int], missing_token: int
    ) -> list[int] | None:
        return self._apply_arithmetic(
            sequence, missing_token, "+", ord("+"), lambda a, b: a + b
        )

    def _apply_simple_subtraction(
        self, sequence: list[int], missing_token: int
    ) -> list[int] | None:
        return self._apply_arithmetic(
            sequence, missing_token, "-", ord("-"), lambda a, b: a - b
        )

    def _apply_simple_multiply(
        self, sequence: list[int], missing_token: int
    ) -> list[int] | None:
        return self._apply_arithmetic(
            sequence, missing_token, "*", ord("*"), lambda a, b: a * b
        )

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

    def apply_with_explanation(
        self, sequence: list[int], missing_token: int = -1
    ) -> dict:
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
                            "support": self.store.trigram_support.get(
                                (left, candidate, right), 0
                            ),
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
                            "support": self.store.bigram_support.get(
                                (left, candidate), 0
                            ),
                        }
                    )
                    filled[index] = candidate

        return {"filled": filled, "rules_hit": rules_hit}


@dataclass
class AutoLearner:
    """自动学习包装器 — 将学习引擎与仿真流自动集成

    每 N ticks 自动从神经元记忆采样，更新规则库，并给予预测奖励。

    用法:
        auto = AutoLearner(network, discovery)
        for t in range(1, ticks + 1):
            snapshot = network.tick(t, dt=dt)
            auto.on_tick(t, snapshot)  # 自动采样 + 学习 + 奖励
    """

    discovery: RuleDiscovery
    store: RuleStore = field(default_factory=RuleStore)
    miner: PatternMiner = field(default_factory=PatternMiner)

    # 配置
    learn_interval: int = 10  # 每 N ticks 执行一次学习
    min_samples: int = 3  # 最低采样数量
    prediction_reward: float = 0.1  # 预测正确能量奖励
    prediction_penalty: float = 0.02  # 预测错误能量惩罚
    max_rules: int = 500  # 规则库容量上限（LRU 淘汰）

    # 统计
    rule_count: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0
    _tick_correct: int = 0  # 当前 tick 的正确预测数
    _tick_total: int = 0  # 当前 tick 的总预测数

    def prediction_accuracy(self) -> float:
        """返回累计预测准确率。

        Returns:
            准确率，范围 [0, 1]，无预测时返回 0.0。
        """
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def on_tick(self, tick: int, snapshot: TickSnapshot, network: "Network") -> dict:
        """每 tick 调用 — 采样、学习、预测奖励

        Args:
            tick: 当前 tick
            snapshot: Network.tick() 返回的 TickSnapshot
            network: 仿真网络实例

        Returns:
            学习报告字典，包含 rule_count、prediction_accuracy 等
        """
        self._tick_correct = 0
        self._tick_total = 0

        is_vectorized = all(
            hasattr(network, attr)
            for attr in ("token_memory", "token_mem_len", "token_mem_ptr", "energy")
        )

        if is_vectorized:
            self._evaluate_predictions_vectorized(network)
            if tick % self.learn_interval == 0:
                self._learn_from_vectorized(network)
        else:
            # 对活跃神经元进行预测验证
            self._evaluate_predictions(network)

            # 按间隔执行学习
            if tick % self.learn_interval == 0:
                self._learn_from_neurons(network)

        report = {
            "rule_count": self.rule_count,
            "prediction_accuracy": self.prediction_accuracy(),
            "tick_predictions": self._tick_total,
            "tick_correct": self._tick_correct,
        }
        return report

    def _learn_from_neurons(self, network: "Network") -> None:
        """从神经元记忆中提取样本并更新规则库"""
        samples: list[list[int]] = []
        expressions: list[str] = []

        for neuron in network.neurons.values():
            if not neuron.alive:
                continue
            # 从记忆中提取 token 序列
            tokens = [event.token_id for event in list(neuron.mem)]
            if len(tokens) >= 3:
                samples.append(tokens)
            # 提取表达式字符串
            raw_expr = "".join(
                chr(t) for t in tokens if ASCII_PRINTABLE_MIN <= t < ASCII_PRINTABLE_MAX
            )
            if "=" in raw_expr:
                expressions.append(raw_expr)

        if len(samples) >= self.min_samples:
            # 更新规则库（N-gram 模式）
            self.store = self.discovery.discover(samples)
            self.rule_count = len(self.store.trigram_support) + len(
                self.store.bigram_support
            )

            # 淘汰老旧规则（LRU 策略的简化版：保留支持度最高的）
            if self.rule_count > self.max_rules:
                self._prune_rules()

        if expressions:
            # 更新符号模式表
            self.miner.fit(expressions)
            self.rule_count += len(self.miner.table)

    def _learn_from_vectorized(self, network: Any) -> None:
        """从向量化引擎 token_memory 中提取样本并更新规则库。"""
        samples: list[list[int]] = []
        expressions: list[str] = []

        alive = np.asarray(network.alive, dtype=np.bool_)
        token_memory = np.asarray(network.token_memory, dtype=np.int32)
        token_mem_len = np.asarray(network.token_mem_len, dtype=np.int32)
        token_mem_ptr = np.asarray(network.token_mem_ptr, dtype=np.int32)

        if token_memory.ndim != 2 or token_memory.shape[0] == 0:
            return

        mem_size = token_memory.shape[1]
        if mem_size <= 0:
            return

        for idx in np.where(alive)[0]:
            length = int(token_mem_len[idx])
            if length <= 0:
                continue
            length = min(length, mem_size)
            ptr = int(token_mem_ptr[idx] % mem_size)
            start = (ptr - length) % mem_size
            row = token_memory[idx]
            if start + length <= mem_size:
                seq = row[start : start + length]
            else:
                split = mem_size - start
                seq = np.concatenate((row[start:], row[: length - split]))
            tokens = [int(x) for x in seq.tolist()]
            if len(tokens) >= 3:
                samples.append(tokens)
            raw_expr = "".join(
                chr(t) for t in tokens if ASCII_PRINTABLE_MIN <= t < ASCII_PRINTABLE_MAX
            )
            if "=" in raw_expr:
                expressions.append(raw_expr)

        if len(samples) >= self.min_samples:
            self.store = self.discovery.discover(samples)
            self.rule_count = len(self.store.trigram_support) + len(
                self.store.bigram_support
            )
            if self.rule_count > self.max_rules:
                self._prune_rules()

        if expressions:
            self.miner.fit(expressions)
            self.rule_count += len(self.miner.table)

    def _evaluate_predictions(self, network: "Network") -> None:
        """对每个活跃神经元评估预测并给予奖励/惩罚"""
        if self.rule_count == 0:
            return  # 规则库为空时跳过

        composer = RuleComposer(store=self.store)

        for neuron in network.neurons.values():
            if not neuron.alive:
                continue
            tokens = [event.token_id for event in list(neuron.mem)]
            if len(tokens) < 4:
                continue

            # 用前 N-1 个 token 预测最后一个
            query_seq = list(tokens[:-1]) + [-1]  # -1 为占位符
            actual = tokens[-1]
            filled = composer.apply(query_seq, missing_token=-1)
            predicted = filled[-1]

            self._tick_total += 1
            self.total_predictions += 1

            if predicted == actual and predicted != -1:
                # 预测正确 — 奖励
                self._tick_correct += 1
                self.correct_predictions += 1
                neuron.energy += self.prediction_reward
                neuron.prediction_hits += 1
            elif predicted != -1:
                # 预测错误 — 轻微惩罚
                neuron.energy = max(0.0, neuron.energy - self.prediction_penalty)

    def _evaluate_predictions_vectorized(self, network: Any) -> None:
        """对向量化网络评估预测并批量执行奖励/惩罚。"""
        if self.rule_count == 0:
            return

        alive = np.asarray(network.alive, dtype=np.bool_)
        token_memory = np.asarray(network.token_memory, dtype=np.int32)
        token_mem_len = np.asarray(network.token_mem_len, dtype=np.int32)
        token_mem_ptr = np.asarray(network.token_mem_ptr, dtype=np.int32)

        if token_memory.ndim != 2 or token_memory.shape[0] == 0:
            return

        mem_size = token_memory.shape[1]
        if mem_size <= 0:
            return

        composer = RuleComposer(store=self.store)
        hit_indices: list[int] = []
        miss_indices: list[int] = []

        for idx in np.where(alive)[0]:
            length = int(token_mem_len[idx])
            if length < 4:
                continue
            length = min(length, mem_size)
            ptr = int(token_mem_ptr[idx] % mem_size)
            start = (ptr - length) % mem_size
            row = token_memory[idx]
            if start + length <= mem_size:
                seq = row[start : start + length]
            else:
                split = mem_size - start
                seq = np.concatenate((row[start:], row[: length - split]))
            tokens = [int(x) for x in seq.tolist()]
            if len(tokens) < 4:
                continue

            query_seq = list(tokens[:-1]) + [-1]
            actual = int(tokens[-1])
            filled = composer.apply(query_seq, missing_token=-1)
            predicted = int(filled[-1])

            self._tick_total += 1
            self.total_predictions += 1

            if predicted == actual and predicted != -1:
                self._tick_correct += 1
                self.correct_predictions += 1
                hit_indices.append(int(idx))
            elif predicted != -1:
                miss_indices.append(int(idx))

        if hit_indices:
            hit_arr = np.asarray(hit_indices, dtype=np.int64)
            network.energy[hit_arr] = network.energy[hit_arr] + self.prediction_reward
            network.prediction_hits[hit_arr] = network.prediction_hits[hit_arr] + 1

        if miss_indices:
            miss_arr = np.asarray(miss_indices, dtype=np.int64)
            network.energy[miss_arr] = np.maximum(
                0.0,
                network.energy[miss_arr] - self.prediction_penalty,
            )

    def _prune_rules(self) -> None:
        """LRU 淘汰低活跃规则，保留 max_rules 以内

        OrderedDict 头部为最久未访问的规则，直接从头部弹出。
        """
        tri_cap = self.max_rules // 2
        while len(self.store.trigram_support) > tri_cap:
            self.store.trigram_support.popitem(last=False)  # 弹出最久未访问
        self.store.trigram_total = sum(self.store.trigram_support.values())

        bi_cap = self.max_rules // 2
        while len(self.store.bigram_support) > bi_cap:
            self.store.bigram_support.popitem(last=False)
        self.store.bigram_total = sum(self.store.bigram_support.values())

        self.rule_count = len(self.store.trigram_support) + len(
            self.store.bigram_support
        )

    def get_summary(self) -> dict:
        """返回学习引擎摘要统计。

        Returns:
            包含 rule_count、prediction_accuracy 等指标的字典。
        """
        return {
            "rule_count": self.rule_count,
            "trigram_rules": len(self.store.trigram_support),
            "bigram_rules": len(self.store.bigram_support),
            "pattern_rules": len(self.miner.table),
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "prediction_accuracy": self.prediction_accuracy(),
        }
