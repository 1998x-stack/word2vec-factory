import pytest
import torch

from w2v_factory.trainer.lr_schedulers import build_linear_warmdown, get_scheduler


def test_linear_warmdown_uses_exact_total_steps():
    decay = build_linear_warmdown(4)
    assert [decay(step) for step in range(6)] == pytest.approx(
        [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    )


def test_scheduler_rejects_unknown_name():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optim = torch.optim.SGD([param], lr=0.1)
    with pytest.raises(ValueError, match="Unsupported lr_schedule"):
        get_scheduler(optim, "liner", 10)


def test_linear_scheduler_reaches_zero_after_planned_steps():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optim = torch.optim.SGD([param], lr=0.1)
    sched = get_scheduler(optim, "linear", 4)
    assert sched is not None

    used_lrs = []
    for _ in range(4):
        used_lrs.append(optim.param_groups[0]["lr"])
        optim.step()
        sched.step()

    assert used_lrs == pytest.approx([0.1, 0.075, 0.05, 0.025])
    assert optim.param_groups[0]["lr"] == pytest.approx(0.0)
