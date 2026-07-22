"""自动学习包装器 — learning 子包的仿真集成部分。

``AutoLearner`` 将规则发现/组合推理与仿真流自动集成，支持标量引擎
（``Network``）与向量化引擎（``VectorizedNetwork``）两条路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from neurogenesis_lab.config import ASCII_PRINTABLE_MIN, ASCII_PRINTABLE_MAX
from neurogenesis_lab.learning.expression import PatternMiner
from neurogenesis_lab.learning.rules import RuleComposer, RuleDiscovery, RuleStore
from neurogenesis_lab.protocols import TickSnapshot

if TYPE_CHECKING:
    from neurogenesis_lab.network import Network


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
            self.rule_count = len(self.store.trigram_support) + len(self.store.bigram_support)

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
            self.rule_count = len(self.store.trigram_support) + len(self.store.bigram_support)
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

        self.rule_count = len(self.store.trigram_support) + len(self.store.bigram_support)

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
