"""单元测试 — learning 模块"""


import pytest
from neurogenesis_lab.encoding import Tokenizer
from neurogenesis_lab.learning import (
    MemoryBank,
    PatternMiner,
    QueryEngine,
    RuleComposer,
    RuleDiscovery,
    RuleStore,
    _parse_expression,
)


class TestParseExpression:
    def test_basic_addition(self) -> None:
        result = _parse_expression("1+2=3")
        assert result == ("1", "+", "2", "3")

    def test_subtraction(self) -> None:
        result = _parse_expression("5-3=2")
        assert result == ("5", "-", "3", "2")

    def test_multiplication(self) -> None:
        result = _parse_expression("2*3=6")
        assert result == ("2", "*", "3", "6")

    def test_division(self) -> None:
        result = _parse_expression("6/2=3")
        assert result == ("6", "/", "2", "3")

    def test_with_spaces(self) -> None:
        result = _parse_expression("1 + 2 = 3")
        assert result == ("1", "+", "2", "3")

    def test_invalid_no_equals(self) -> None:
        assert _parse_expression("1+2") is None

    def test_invalid_too_short(self) -> None:
        assert _parse_expression("a=b") is None

    def test_question_mark(self) -> None:
        result = _parse_expression("1+2=?")
        assert result == ("1", "+", "2", "?")


class TestMemoryBank:
    def test_add_and_retrieve(self) -> None:
        bank = MemoryBank()
        bank.add("1+1=2")
        bank.add_many(["2+2=4", "3+3=6"])
        assert len(bank.expressions) == 3


class TestPatternMiner:
    def test_fit_and_lookup(self) -> None:
        miner = PatternMiner()
        miner.fit(["1+1=2", "1+2=3"])
        assert miner.lookup("1", "+", "1") == "2"
        assert miner.lookup("1", "+", "2") == "3"
        assert miner.lookup("9", "+", "9") is None

    def test_support_count(self) -> None:
        miner = PatternMiner()
        miner.fit(["1+1=2", "1+1=2", "1+2=3"])
        assert miner.support_count[("1", "+", "1")] == 2

    def test_confidence(self) -> None:
        miner = PatternMiner()
        miner.fit(["1+1=2", "1+1=2", "1+2=3"])
        conf = miner.confidence("1", "+", "1")
        assert conf > 0.0
        assert conf <= 1.0


class TestQueryEngine:
    def _setup_engine(self) -> QueryEngine:
        miner = PatternMiner()
        miner.fit(["1+1=2", "1+2=3", "2+2=4"])
        return QueryEngine(miner=miner)

    def test_known_query(self) -> None:
        engine = self._setup_engine()
        assert engine.query("1+1=?") == "2"
        assert engine.query("2+2=?") == "4"

    def test_unknown_query(self) -> None:
        engine = self._setup_engine()
        assert engine.query("9+9=?") == "unknown"

    def test_invalid_expression(self) -> None:
        engine = self._setup_engine()
        assert engine.query("hello") == "unknown"

    def test_query_with_explanation(self) -> None:
        engine = self._setup_engine()
        result = engine.query_with_explanation("1+1=?")
        assert result["answer"] == "2"
        assert result["confidence"] > 0
        assert len(result["rules_hit"]) > 0

    def test_unknown_with_explanation(self) -> None:
        engine = self._setup_engine()
        result = engine.query_with_explanation("9+9=?")
        assert result["answer"] == "unknown"
        assert result["confidence"] == 0.0


class TestRuleStore:
    def test_trigram_confidence(self) -> None:
        store = RuleStore(
            trigram_support={(1, 2, 3): 5, (4, 5, 6): 3},
            bigram_support={(1, 2): 4},
        )
        conf = store.trigram_confidence((1, 2, 3))
        assert conf == pytest.approx(5 / 8)

    def test_top_candidates(self) -> None:
        store = RuleStore(
            trigram_support={(1, 10, 3): 5, (1, 20, 3): 3, (1, 30, 3): 1},
            bigram_support={},
        )
        top = store.top_candidates_middle(1, 3, k=2)
        assert len(top) == 2
        assert top[0][0] == 10  # 最高支持度

    def test_most_common_next(self) -> None:
        store = RuleStore(
            trigram_support={},
            bigram_support={(1, 2): 5, (1, 3): 2},
        )
        assert store.most_common_next(1) == 2


class TestRuleDiscovery:
    def test_discover_filters_by_min_support(self) -> None:
        samples = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
        store = RuleDiscovery(min_support=2).discover(samples)
        assert (1, 2, 3) in store.trigram_support
        assert (4, 5, 6) not in store.trigram_support

    def test_discover_bigrams(self) -> None:
        samples = [[1, 2], [1, 2], [3, 4]]
        store = RuleDiscovery(min_support=2).discover(samples)
        assert (1, 2) in store.bigram_support

    def test_discover_uses_bridge(self, monkeypatch) -> None:
        calls = {"count": 0}

        def _fake_mine(samples: list[list[int]], min_support: int):
            calls["count"] += 1
            return {(1, 2, 3): 2}, {(1, 2): 3}, "python-fallback"

        monkeypatch.setattr(
            "neurogenesis_lab.learning.rules.mine_pattern_counts_safe", _fake_mine
        )

        store = RuleDiscovery(min_support=2).discover([[1, 2, 3], [1, 2, 3]])
        assert calls["count"] == 1
        assert store.trigram_support == {(1, 2, 3): 2}
        assert store.bigram_support == {(1, 2): 3}


class TestRuleComposer:
    def _setup_composer(self) -> RuleComposer:
        tokenizer = Tokenizer()
        samples = [
            tokenizer.encode("1+1=2"),
            tokenizer.encode("1+2=3"),
            tokenizer.encode("2+1=3"),
        ]
        store = RuleDiscovery(min_support=2).discover(samples)
        return RuleComposer(store=store)

    def test_fill_addition(self) -> None:
        tokenizer = Tokenizer()
        composer = self._setup_composer()
        query = tokenizer.encode("1+3=?")
        missing = tokenizer.encode("?")[0]
        masked = [t if t != missing else -1 for t in query]
        filled = composer.apply(masked, missing_token=-1)
        answer = "".join(tokenizer.decode_token(t) for t in filled)
        assert answer == "1+3=4"

    def test_fill_subtraction(self) -> None:
        store = RuleStore(trigram_support={}, bigram_support={})
        composer = RuleComposer(store=store)
        tokenizer = Tokenizer()
        query = tokenizer.encode("5-3=?")
        missing = tokenizer.encode("?")[0]
        masked = [t if t != missing else -1 for t in query]
        filled = composer.apply(masked, missing_token=-1)
        answer = "".join(tokenizer.decode_token(t) for t in filled)
        assert answer == "5-3=2"

    def test_fill_multiplication(self) -> None:
        store = RuleStore(trigram_support={}, bigram_support={})
        composer = RuleComposer(store=store)
        tokenizer = Tokenizer()
        query = tokenizer.encode("2*4=?")
        missing = tokenizer.encode("?")[0]
        masked = [t if t != missing else -1 for t in query]
        filled = composer.apply(masked, missing_token=-1)
        answer = "".join(tokenizer.decode_token(t) for t in filled)
        assert answer == "2*4=8"

    def test_apply_with_explanation(self) -> None:
        tokenizer = Tokenizer()
        composer = self._setup_composer()
        query = tokenizer.encode("1+3=?")
        missing = tokenizer.encode("?")[0]
        masked = [t if t != missing else -1 for t in query]
        result = composer.apply_with_explanation(masked, missing_token=-1)
        assert "filled" in result
        assert "rules_hit" in result
        assert len(result["rules_hit"]) > 0
        answer = "".join(tokenizer.decode_token(t) for t in result["filled"])
        assert answer == "1+3=4"

    def test_unknown_returns_gracefully(self) -> None:
        """无规则匹配时不崩溃"""
        store = RuleStore(trigram_support={}, bigram_support={})
        composer = RuleComposer(store=store)
        result = composer.apply([1, -1, 3], missing_token=-1)
        assert isinstance(result, list)


class TestRicherSamples:
    """验证非加法样本的泛化与失败模式"""

    def test_subtraction_samples(self) -> None:
        miner = PatternMiner()
        miner.fit(["5-3=2", "7-2=5", "9-4=5"])
        engine = QueryEngine(miner=miner)
        assert engine.query("5-3=?") == "2"
        assert engine.query("7-2=?") == "5"
        assert engine.query("1-1=?") == "unknown"  # 未见过

    def test_mixed_operations(self) -> None:
        miner = PatternMiner()
        miner.fit(["1+1=2", "2*3=6", "4-1=3"])
        engine = QueryEngine(miner=miner)
        assert engine.query("1+1=?") == "2"
        assert engine.query("2*3=?") == "6"
        assert engine.query("4-1=?") == "3"
