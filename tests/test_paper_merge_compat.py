"""Check both planning protocols and shared model/optimizer registration."""
from w2v_factory.config import Cfg, DataCfg, ModelCfg
from w2v_factory.experiments.paper_plan import build_plan as build_paper_plan, load_suite
from w2v_factory.experiments.plan import build_plan as build_grid_plan
from w2v_factory.factory import make_model, make_optimizer


def test_both_planners_retain_distinct_yaml_contracts():
    grid, _ = build_grid_plan("configs/paper1301_ablations.yaml")
    cases = build_paper_plan(load_suite("configs/experiments/smoke.yaml"))
    assert grid and len(cases) == 5


def test_model_registry_and_optimizer_factory_share_common_engine():
    model = make_model(
        "cbow", vocab_size=5, dim=4, out_vocab_size=4, share_io=False, pooling="mean"
    )
    optimizer = make_optimizer("sgd", params=model.parameters(), lr=0.025)
    assert model.in_embed.weight.shape == (5, 4)
    assert optimizer.param_groups[0]["lr"] == 0.025


def test_subsample_formula_and_window_config():
    assert Cfg().DATA.subsample_formula == "legacy"
    assert DataCfg(subsample_formula="mikolov_1310").subsample_formula == "mikolov_1310"
    assert ModelCfg(window_sampling="fixed").window_sampling == "fixed"
