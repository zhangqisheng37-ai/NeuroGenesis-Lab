from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Event:
    """不可变事件 — 神经网络中传播的基本信息单元。

    Attributes:
        token_id: 编码后的 token ID。
        t: 事件时间戳（tick 编号）。
        src_id: 源/目标神经元 ID。
        meta: 附加元数据字典，如 weight、symbol 等。
    """

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
    """不可变 token 序列 — 编码后的表达式表示。

    Attributes:
        tokens: token ID 列表。
    """

    tokens: list[int]

    @classmethod
    def from_expression(cls, expr: str, encoder: "Encoder") -> "Sequence":
        """从表达式字符串创建 Sequence。

        Args:
            expr: 原始表达式字符串。
            encoder: 编码器实例。

        Returns:
            编码后的 Sequence 实例。
        """
        return cls(tokens=encoder.encode_expr(expr))


class Vocabulary:
    """符号词表 — 管理符号与 token ID 的双向映射。

    Args:
        symbols: 符号字符串可迭代对象，按顺序分配 ID（从 0 开始）。

    Example:
        >>> vocab = Vocabulary(["a", "b", "+", "="])
        >>> vocab.encode_symbol("a")
        0
    """

    def __init__(self, symbols: Iterable[str]) -> None:
        self._symbols = list(symbols)
        self._symbol_to_id = {symbol: index for index, symbol in enumerate(self._symbols)}
        self._id_to_symbol = {index: symbol for index, symbol in enumerate(self._symbols)}

    def encode_symbol(self, symbol: str) -> int:
        """将符号编码为 token ID。

        Args:
            symbol: 待编码的符号字符串。

        Returns:
            对应的 token ID。

        Raises:
            ValueError: 符号不在词表中时抛出。
        """
        if symbol not in self._symbol_to_id:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self._symbol_to_id[symbol]

    def decode_token(self, token_id: int) -> str:
        """将 token ID 解码为符号字符串。

        Args:
            token_id: 待解码的 token ID。

        Returns:
            对应的符号字符串。

        Raises:
            ValueError: token ID 不在词表中时抛出。
        """
        if token_id not in self._id_to_symbol:
            raise ValueError(f"Unknown token id: {token_id}")
        return self._id_to_symbol[token_id]


class Tokenizer:
    """分词器 — 将文本逐字符编码为 token ID 序列。

    若提供 Vocabulary 则使用词表映射，否则回退到 ``ord()``/``chr()``。

    Args:
        vocabulary: 可选的词表实例。
    """

    def __init__(self, vocabulary: Vocabulary | None = None) -> None:
        self._vocabulary = vocabulary

    def encode(self, text: str) -> list[int]:
        """将文本逐字符编码为 token ID 列表。

        Args:
            text: 待编码的文本字符串。

        Returns:
            token ID 列表，长度等于 text 字符数。
        """
        return [self.encode_symbol(symbol) for symbol in text]

    def encode_symbol(self, symbol: str) -> int:
        """将单个符号编码为 token ID。

        Args:
            symbol: 单个字符。

        Returns:
            token ID，有词表时查表，否则返回 ``ord(symbol)``。
        """
        if self._vocabulary is not None:
            return self._vocabulary.encode_symbol(symbol)
        return ord(symbol)

    def decode_token(self, token_id: int) -> str:
        """将 token ID 解码为符号字符串。

        Args:
            token_id: 待解码的 token ID。

        Returns:
            符号字符串，有词表时查表，否则返回 ``chr(token_id)``。
        """
        if self._vocabulary is not None:
            return self._vocabulary.decode_token(token_id)
        return chr(token_id)


class Encoder:
    """编码器 — 将表达式字符串编码为 token 序列或事件列表。

    Args:
        tokenizer: 可选的分词器实例，默认使用无词表的 Tokenizer。

    Example:
        >>> encoder = Encoder()
        >>> tokens = encoder.encode_expr("1+2=3")
        >>> events = encoder.to_events(tokens, src_id="input")
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or Tokenizer()

    def encode_expr(self, expr: str) -> list[int]:
        """将表达式字符串编码为 token ID 列表。

        Args:
            expr: 表达式字符串。

        Returns:
            token ID 列表。
        """
        return self._tokenizer.encode(expr)

    def to_events(self, tokens: Iterable[int], src_id: str = "encoder") -> list[Event]:
        """将 token 序列转换为 Event 列表。

        每个 token 生成一个 Event，时间戳从 1 递增。

        Args:
            tokens: token ID 可迭代对象。
            src_id: 事件源 ID，默认 ``"encoder"``。

        Returns:
            Event 列表，长度等于 token 数量。
        """
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
