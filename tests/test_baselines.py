"""
tests/test_baselines.py
-----------------------
Unit tests for all baseline weighting methods.

Tests:
    1. Interface conformance (all return scalar + dict)
    2. Kendall log_var explosion
    3. UW-SO no learnable parameters
    4. Static sum correctness
    5. PCGrad conflict detection
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectra.baselines.static import StaticWeighter
from spectra.baselines.kendall import KendallWeighter
from spectra.baselines.uwso import UWSOWeighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.baselines.ntkmtl import NTKMTLWeighter


ALL_WEIGHTERS = [
    ("static", StaticWeighter),
    ("kendall", KendallWeighter),
    ("uwso", UWSOWeighter),
    ("pcgrad", PCGradWeighter),
    ("ntkmtl", NTKMTLWeighter),
]


class TestInterfaceConformance:
    """All methods must return (scalar, dict) from forward()."""

    @pytest.mark.parametrize("name,cls", ALL_WEIGHTERS)
    def test_forward_returns_correct_types(self, name, cls):
        weighter = cls(num_tasks=3)
        losses = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        total, metrics = weighter(losses, sync_ddp=False)

        assert total.dim() == 0, f"{name}: total_loss should be scalar"
        assert isinstance(metrics, dict), f"{name}: metrics should be dict"
        assert total.requires_grad, f"{name}: total_loss should require grad"


class TestKendallExplosion:
    """Kendall log_vars should drift on high-scale-gap tasks."""

    def test_log_var_drift(self):
        weighter = KendallWeighter(num_tasks=3)
        optimizer = torch.optim.Adam(weighter.parameters(), lr=0.05)

        for _ in range(300):
            losses = torch.tensor([3000.0, 2.0, 0.5], requires_grad=True)
            total, _ = weighter(losses, sync_ddp=False)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        # log_var for the dominant task should have drifted significantly
        log_var_0 = weighter.log_vars[0].item()
        assert log_var_0 > 3.0, (
            f"Kendall log_var[0] should explode, got {log_var_0:.2f}"
        )


class TestUWSO:
    """UW-SO has zero learnable parameters."""

    def test_no_parameters(self):
        weighter = UWSOWeighter(num_tasks=3)
        n_params = sum(1 for _ in weighter.parameters())
        assert n_params == 0, f"UW-SO should have 0 params, got {n_params}"

    def test_weights_update(self):
        weighter = UWSOWeighter(num_tasks=2)
        losses1 = torch.tensor([100.0, 1.0], requires_grad=True)
        _, m1 = weighter(losses1, sync_ddp=False)

        # After seeing losses, weights should reflect scale
        w0 = m1["uwso/weight_0"]
        w1 = m1["uwso/weight_1"]
        # Task 0 (loss=100) should get lower weight than task 1 (loss=1)
        assert w0 < w1, f"UW-SO weight ordering wrong: w0={w0}, w1={w1}"


class TestStatic:
    """Static sum should be exact."""

    def test_equal_weights_sum(self):
        weighter = StaticWeighter(num_tasks=3)
        losses = torch.tensor([3.0, 2.0, 1.0], requires_grad=True)
        total, _ = weighter(losses, sync_ddp=False)
        expected = (3.0 + 2.0 + 1.0) / 3.0
        assert abs(total.item() - expected) < 1e-5

    def test_custom_weights(self):
        weighter = StaticWeighter(num_tasks=3, weights=[0.5, 0.3, 0.2])
        losses = torch.tensor([10.0, 10.0, 10.0], requires_grad=True)
        total, _ = weighter(losses, sync_ddp=False)
        expected = 0.5 * 10 + 0.3 * 10 + 0.2 * 10
        assert abs(total.item() - expected) < 1e-5


class TestPCGrad:
    """PCGrad should detect and resolve gradient conflicts."""

    def test_projections_run(self):
        weighter = PCGradWeighter(num_tasks=2)
        # Simple model for gradient computation
        model = torch.nn.Linear(5, 1, bias=False)
        x = torch.randn(4, 5)

        loss1 = (model(x).mean()) ** 2
        loss2 = (-(model(x).mean())) ** 2

        metrics = weighter.backward_and_project(
            [loss1, loss2],
            list(model.parameters()),
        )
        assert "pcgrad/total_conflicts" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
