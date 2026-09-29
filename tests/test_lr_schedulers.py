import pytest
import torch

from w2v_factory.trainer.lr_schedulers import TokenProgressScheduler, get_scheduler


def test_token_scheduler_uses_source_progress():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optim = torch.optim.SGD([param], lr=0.1)
    sched = TokenProgressScheduler(optim, total_tokens=100)

    expected = [(0, 0.1), (25, 0.075), (50, 0.05), (75, 0.025), (100, 0.0)]
    for progress, lr in expected:
        sched.set_progress(progress)
        assert optim.param_groups[0]["lr"] == pytest.approx(lr)


def test_token_scheduler_rejects_backward_or_out_of_range_progress():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optim = torch.optim.SGD([param], lr=0.1)
    sched = TokenProgressScheduler(optim, total_tokens=10)
    sched.set_progress(5)
    with pytest.raises(ValueError, match="monotonic"):
        sched.set_progress(4)
    with pytest.raises(ValueError, match="outside"):
        sched.set_progress(11)


def test_scheduler_rejects_unknown_name():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optim = torch.optim.SGD([param], lr=0.1)
    with pytest.raises(ValueError, match="Unsupported lr_schedule"):
        get_scheduler(optim, "liner", 10)


def test_none_scheduler_does_not_change_lr():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optim = torch.optim.SGD([param], lr=0.1)
    assert get_scheduler(optim, "none", 100) is None
    assert optim.param_groups[0]["lr"] == pytest.approx(0.1)
