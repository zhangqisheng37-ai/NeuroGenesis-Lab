from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from neurogenesis_lab.core_bridge import mine_pattern_counts_safe

if TYPE_CHECKING:
    from neurogenesis_lab.network import Network


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

    def top_candidates_middle(
        self, left: int, right: int, k: int = 3
    ) -> list[tuple[int, int]]:
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
        filtered_trigrams, filtered_bigrams, _backend = mine_pattern_counts_safe(
            samples, self.min_support
        )
        return RuleStore(
            trigram_support=filtered_trigrams, bigram_support=filtered_bigrams
        )


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

    def _apply_arithmetic(
        self,
        sequence: list[int],
        missing_token: int,
        op: str,
        op_char: int,
        calc: callable,
    ) -> list[int] | None:
        """通用算术运算填充方法"""
        missing_indices = [
            index for index, token in enumerate(sequence) if token == missing_token
        ]
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
        """返回累计预测准确率"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def on_tick(self, tick: int, snapshot: dict, network: "Network") -> dict:
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
            raw_expr = "".join(chr(t) for t in tokens if 32 <= t < 127)
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

    def _prune_rules(self) -> None:
        """淘汰低支持度规则，保留 max_rules 以内"""
        # 淘汰 trigram
        if len(self.store.trigram_support) > self.max_rules // 2:
            sorted_tri = sorted(
                self.store.trigram_support.items(), key=lambda x: x[1], reverse=True
            )
            keep = self.max_rules // 2
            self.store.trigram_support = dict(sorted_tri[:keep])
            self.store.trigram_total = sum(self.store.trigram_support.values())

        # 淘汰 bigram
        if len(self.store.bigram_support) > self.max_rules // 2:
            sorted_bi = sorted(
                self.store.bigram_support.items(), key=lambda x: x[1], reverse=True
            )
            keep = self.max_rules // 2
            self.store.bigram_support = dict(sorted_bi[:keep])
            self.store.bigram_total = sum(self.store.bigram_support.values())

        self.rule_count = len(self.store.trigram_support) + len(
            self.store.bigram_support
        )

    def get_summary(self) -> dict:
        """返回学习引擎摘要"""
        return {
            "rule_count": self.rule_count,
            "trigram_rules": len(self.store.trigram_support),
            "bigram_rules": len(self.store.bigram_support),
            "pattern_rules": len(self.miner.table),
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "prediction_accuracy": self.prediction_accuracy(),
        }
