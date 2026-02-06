import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Tokenizer
from neurogenesis_lab.learning import RuleComposer, RuleDiscovery


def test_rule_discovery_and_composer_fill_missing_token() -> None:
    tokenizer = Tokenizer()
    samples = [
        tokenizer.encode("1+1=2"),
        tokenizer.encode("1+2=3"),
        tokenizer.encode("2+1=3"),
    ]

    store = RuleDiscovery(min_support=2).discover(samples)
    composer = RuleComposer(store=store)

    query = tokenizer.encode("1+3=?")
    missing_token = tokenizer.encode("?")[0]
    masked = [token if token != missing_token else -1 for token in query]
    filled = composer.apply(masked, missing_token=-1)
    answer = "".join(tokenizer.decode_token(token) for token in filled)

    assert len(store.bigram_support) > 0
    assert len(store.trigram_support) > 0
    assert answer == "1+3=4"
