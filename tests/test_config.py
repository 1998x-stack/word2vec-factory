import pytest

from w2v_factory.config import load_cfg


def test_include_merge(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("SEED: 7\nMODEL:\n  dim: 100\n  loss: ns\nTRAIN:\n  lr: 0.01\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"INCLUDE: {base}\nMODEL:\n  loss: hs\n")
    cfg = load_cfg(str(child))
    assert cfg.SEED == 7
    assert cfg.MODEL.dim == 100
    assert cfg.MODEL.loss == "hs"
    assert cfg.TRAIN.lr == 0.01


def test_subsample_coercion_through_load(tmp_path):
    fnone = tmp_path / "a.yaml"
    fnone.write_text("DATA:\n  subsample_t: null\n")
    assert load_cfg(str(fnone)).DATA.subsample_t is None

    fstr = tmp_path / "b.yaml"
    fstr.write_text("DATA:\n  subsample_t: '1e-5'\n")
    assert load_cfg(str(fstr)).DATA.subsample_t == pytest.approx(1e-5)
