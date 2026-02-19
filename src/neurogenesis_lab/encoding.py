from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Event:
    token_id: int
    t: int
    src_id: str
    meta: dict

    def __lt__(self, other: "Event") -> bool:
        """支持 heapq 比较，确保 EventQueue 稳定性"""
        if not isinstance(other, Event):
            return NotImplemented
        return (self.t, id(self)) < (other.t, id(other))


@dataclass(frozen=True)
class Sequence:
    tokens: list[int]

    @classmethod
    def from_expression(cls, expr: str, encoder: "Encoder") -> "Sequence":
        return cls(tokens=encoder.encode_expr(expr))


class Vocabulary:
    def __init__(self, symbols: Iterable[str]) -> None:
        self._symbols = list(symbols)
        self._symbol_to_id = {symbol: index for index, symbol in enumerate(self._symbols)}
        self._id_to_symbol = {index: symbol for index, symbol in enumerate(self._symbols)}

    def encode_symbol(self, symbol: str) -> int:
        if symbol not in self._symbol_to_id:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self._symbol_to_id[symbol]

    def decode_token(self, token_id: int) -> str:
        if token_id not in self._id_to_symbol:
            raise ValueError(f"Unknown token id: {token_id}")
        return self._id_to_symbol[token_id]


class Tokenizer:
    def __init__(self, vocabulary: Vocabulary | None = None) -> None:
        self._vocabulary = vocabulary

    def encode(self, text: str) -> list[int]:
        return [self.encode_symbol(symbol) for symbol in text]

    def encode_symbol(self, symbol: str) -> int:
        if self._vocabulary is not None:
            return self._vocabulary.encode_symbol(symbol)
        return ord(symbol)

    def decode_token(self, token_id: int) -> str:
        if self._vocabulary is not None:
            return self._vocabulary.decode_token(token_id)
        return chr(token_id)


class Encoder:
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or Tokenizer()

    def encode_expr(self, expr: str) -> list[int]:
        return self._tokenizer.encode(expr)

    def to_events(self, tokens: Iterable[int], src_id: str = "encoder") -> list[Event]:
        events: list[Event] = []
        for index, token_id in enumerate(tokens, start=1):
            events.append(
                Event(
                    token_id=token_id,
                    t=index,
                    src_id=src_id,
                    meta={"symbol": self._tokenizer.decode_token(token_id), "pos": index},
                )
            )
        return events
