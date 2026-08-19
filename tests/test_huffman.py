from w2v_factory.data.huffman import build_huffman_codes


def _is_prefix(a, b):
    return len(a) <= len(b) and b[: len(a)] == a


def test_inner_nodes_and_path_lengths():
    counts = [3, 2, 5, 1, 7]
    paths, codes = build_huffman_codes(counts)
    V = len(counts)
    assert len(paths) == V and len(codes) == V
    for p, c in zip(paths, codes):
        assert len(p) == len(c)  # one code bit per tree edge
        assert len(p) >= 1  # every word has a non-empty path


def test_unique_prefix_free_codes():
    counts = [3, 2, 5, 1, 7, 4, 6]
    _, codes = build_huffman_codes(counts)
    # Words at the same branch share path nodes; only the 0/1 codes differ.
    # Every leaf must map to a unique root-to-leaf code string, and no code
    # may be a prefix of another (a full binary tree is prefix-free).
    code_tuples = [tuple(c) for c in codes]
    assert len(set(code_tuples)) == len(counts)
    for a in code_tuples:
        for b in code_tuples:
            if a != b:
                assert not _is_prefix(a, b)


def test_codes_are_binary():
    counts = [3, 2, 5, 1, 7]
    _, codes = build_huffman_codes(counts)
    flat = [b for c in codes for b in c]
    assert set(flat) <= {0, 1}


def test_deterministic():
    counts = [3, 2, 5, 1, 7, 4, 6]
    p1, c1 = build_huffman_codes(list(counts))
    p2, c2 = build_huffman_codes(list(counts))
    assert p1 == p2 and c1 == c2