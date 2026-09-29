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


def test_include_path_is_relative_to_child_config(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text("SEED: 11\nMODEL:\n  dim: 64\n", encoding="utf-8")
    child = cfg_dir / "child.yaml"
    child.write_text("INCLUDE: base.yaml\nMODEL:\n  loss: hs\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    cfg = load_cfg(str(child))
    assert cfg.SEED == 11
    assert cfg.MODEL.dim == 64
    assert cfg.MODEL.loss == "hs"


def test_unknown_config_keys_fail_fast(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("MODEL:\n  dmi: 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unknown MODEL config key.*dmi"):
        load_cfg(str(bad))


def test_cyclic_include_is_rejected(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("INCLUDE: b.yaml\n", encoding="utf-8")
    b.write_text("INCLUDE: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic config INCLUDE"):
        load_cfg(str(a))
