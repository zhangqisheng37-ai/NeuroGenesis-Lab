
from neurogenesis_lab.encoding import Encoder


def test_encode_expr_and_event_order() -> None:
    encoder = Encoder()
    expr = "1+1=2"

    tokens = encoder.encode_expr(expr)
    events = encoder.to_events(tokens, src_id="test-src")

    assert tokens == [ord(symbol) for symbol in expr]
    assert len(events) == len(tokens)
    assert [event.t for event in events] == list(range(1, len(expr) + 1))
    assert [event.src_id for event in events] == ["test-src"] * len(events)
    assert [event.meta["symbol"] for event in events] == list(expr)
