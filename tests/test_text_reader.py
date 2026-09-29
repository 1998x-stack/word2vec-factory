from w2v_factory.data.text_reader import iter_tokens
from w2v_factory.registry import TOKENIZER_REG


def test_whitespace_alias_is_registered():
    simple = TOKENIZER_REG.get("simple")
    whitespace = TOKENIZER_REG.get("whitespace")
    assert whitespace is simple
    assert whitespace("Hello   WORLD", lowercase=True) == ["hello", "world"]


def test_iter_tokens_accepts_whitespace_alias(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Alpha Beta\n", encoding="utf-8")
    rows = list(iter_tokens([str(corpus)], True, "whitespace", 100))
    assert rows == [["alpha", "beta"]]
