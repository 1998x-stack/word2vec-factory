from w2v_factory.data.vocab import build_vocab


def test_build_vocab_min_count_and_ordering():
    stream = [["a", "b", "a", "c"], ["a", "b"], ["c", "d"]]
    v = build_vocab(stream, min_count=2, max_vocab=10)
    # counts: a=3, b=2, c=2, d=1 -> d drops below min_count
    assert v.itos == ["a", "b", "c"]
    assert v.stoi == {"a": 0, "b": 1, "c": 2}
    assert v.counts == [3, 2, 2]
    assert v.size == 3
    assert v.total_tokens == 8


def test_max_vocab_limits():
    stream = [["a"], ["b"], ["c"], ["d"]]
    v = build_vocab(stream, min_count=1, max_vocab=2)
    assert v.size == 2
