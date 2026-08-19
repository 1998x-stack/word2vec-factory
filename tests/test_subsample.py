import numpy as np
import pytest

from w2v_factory.data.subsample import _coerce_t, compute_discard_probs


@pytest.mark.parametrize("v", [None, "null", "None", "false", "", "0"])
def test_coerce_null_variants(v):
    assert _coerce_t(v) is None


def test_coerce_numeric():
    assert _coerce_t(1e-5) == pytest.approx(1e-5)
    assert _coerce_t("1e-5") == pytest.approx(1e-5)
    assert _coerce_t("5e-4") == pytest.approx(5e-4)


def test_discard_disabled_returns_zeros():
    probs = compute_discard_probs([10, 5], 100, None)
    assert probs.shape == (2,)
    assert (probs == 0).all()


def test_discard_enabled_shapes_and_bounds():
    counts = [1000, 500, 100, 50, 10, 1]
    probs = compute_discard_probs(counts, sum(counts), 1e-5)
    assert probs.shape == (len(counts),)
    assert probs.dtype == np.float32
    assert (probs >= 0).all() and (probs <= 1).all()


def test_discard_string_t_matches_float_t():
    counts = [9000, 900, 100]
    total = 10000
    p_num = compute_discard_probs(counts, total, 1e-5)
    p_str = compute_discard_probs(counts, total, "1e-5")
    np.testing.assert_allclose(p_num, p_str)
