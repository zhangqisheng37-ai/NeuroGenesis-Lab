"""单元测试 — encoding 模块"""

import pytest
from neurogenesis_lab.encoding import Encoder, Event, Sequence, Tokenizer, Vocabulary


class TestVocabulary:
    def test_encode_known_symbol(self) -> None:
        vocab = Vocabulary(["a", "b", "c"])
        assert vocab.encode_symbol("a") == 0
        assert vocab.encode_symbol("c") == 2

    def test_encode_unknown_symbol_raises(self) -> None:
        vocab = Vocabulary(["a", "b"])
        with pytest.raises(ValueError, match="Unknown symbol: z"):
            vocab.encode_symbol("z")

    def test_decode_known_token(self) -> None:
        vocab = Vocabulary(["x", "y"])
        assert vocab.decode_token(0) == "x"
        assert vocab.decode_token(1) == "y"

    def test_decode_unknown_token_raises(self) -> None:
        vocab = Vocabulary(["x"])
        with pytest.raises(ValueError, match="Unknown token id: 99"):
            vocab.decode_token(99)


class TestTokenizer:
    def test_encode_without_vocabulary(self) -> None:
        tokenizer = Tokenizer()
        tokens = tokenizer.encode("AB")
        assert tokens == [65, 66]

    def test_encode_with_vocabulary(self) -> None:
        vocab = Vocabulary(["1", "+", "="])
        tokenizer = Tokenizer(vocabulary=vocab)
        assert tokenizer.encode_symbol("1") == 0
        assert tokenizer.encode_symbol("+") == 1

    def test_decode_without_vocabulary(self) -> None:
        tokenizer = Tokenizer()
        assert tokenizer.decode_token(65) == "A"

    def test_roundtrip(self) -> None:
        tokenizer = Tokenizer()
        text = "1+2=3"
        tokens = tokenizer.encode(text)
        decoded = "".join(tokenizer.decode_token(t) for t in tokens)
        assert decoded == text


class TestEncoder:
    def test_encode_expr(self) -> None:
        encoder = Encoder()
        tokens = encoder.encode_expr("1+1=2")
        assert tokens == [ord(c) for c in "1+1=2"]

    def test_to_events_order(self) -> None:
        encoder = Encoder()
        tokens = encoder.encode_expr("abc")
        events = encoder.to_events(tokens, src_id="src")
        assert len(events) == 3
        assert [e.t for e in events] == [1, 2, 3]
        assert all(e.src_id == "src" for e in events)

    def test_to_events_meta_has_symbol(self) -> None:
        encoder = Encoder()
        tokens = encoder.encode_expr("1+2")
        events = encoder.to_events(tokens, src_id="s")
        assert events[0].meta["symbol"] == "1"
        assert events[1].meta["symbol"] == "+"
        assert events[2].meta["symbol"] == "2"

    def test_empty_expression(self) -> None:
        encoder = Encoder()
        tokens = encoder.encode_expr("")
        assert tokens == []
        events = encoder.to_events(tokens, src_id="s")
        assert events == []


class TestSequence:
    def test_from_expression(self) -> None:
        encoder = Encoder()
        seq = Sequence.from_expression("1+1", encoder)
        assert seq.tokens == [ord("1"), ord("+"), ord("1")]


class TestEvent:
    def test_event_is_frozen(self) -> None:
        event = Event(token_id=1, t=0, src_id="s", meta={})
        with pytest.raises(AttributeError):
            event.token_id = 2  # type: ignore
