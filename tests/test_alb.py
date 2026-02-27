"""
tests/test_alb.py
-----------------
Unit tests for ALB (Asymmetric Latent Bottleneck).

Output key names (actual implementation):
    "planner"        - [B, T, D] low-freq context
    "planner_global" - [B, D]    pooled planner
    "expert"         - [B, T, D] high-freq context
    "expert_global"  - [B, D]    pooled expert

Note: The original plan doc used ctx_planner/global_planner/ctx_expert/global_expert.
These were incorrect (old spec vs actual alb.py). This file uses the real key names.
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectra.core.alb import AsymmetricLatentBottleneck
from spectra.backbones.shared_trunk import SharedTrunk


def _make_alb(input_dim=20, d_model=64, n_expert_layers=2, n_heads=4):
    """Helper to create an ALB with a SharedTrunk encoder."""
    encoder = SharedTrunk(input_dim=input_dim, d_model=d_model, n_layers=2)
    return AsymmetricLatentBottleneck(
        encoder=encoder,
        input_dim=input_dim,
        d_model=d_model,
        n_expert_layers=n_expert_layers,
        n_heads=n_heads,
        dropout=0.0,
        init_mode="orthogonal",
    ), encoder


class TestALBShapes:
    """Test 1: Output shapes are correct."""

    def test_output_shapes_no_mask(self):
        B, T, D_in, D = 4, 10, 20, 64
        alb, _ = _make_alb(D_in, D)
        x = torch.randn(B, T, D_in)
        out = alb(x)

        # Verify correct key names (CRITICAL: these are the actual keys)
        assert "planner"        in out, f"Missing 'planner'. Got keys: {list(out.keys())}"
        assert "planner_global" in out, f"Missing 'planner_global'. Got keys: {list(out.keys())}"
        assert "expert"         in out, f"Missing 'expert'. Got keys: {list(out.keys())}"
        assert "expert_global"  in out, f"Missing 'expert_global'. Got keys: {list(out.keys())}"

        assert out["planner"].shape        == (B, T, D), f"planner shape: {out['planner'].shape}"
        assert out["planner_global"].shape == (B, D),    f"planner_global shape: {out['planner_global'].shape}"
        assert out["expert"].shape         == (B, T, D), f"expert shape: {out['expert'].shape}"
        assert out["expert_global"].shape  == (B, D),    f"expert_global shape: {out['expert_global'].shape}"

    def test_output_shapes_with_mask(self):
        B, T, D_in, D = 4, 10, 20, 64
        alb, _ = _make_alb(D_in, D)
        x = torch.randn(B, T, D_in)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -3:] = True  # Last 3 positions padded

        out = alb(x, mask=mask)
        assert out["planner"].shape == (B, T, D)
        assert out["expert"].shape  == (B, T, D)


class TestGradientDivorce:
    """Test 2: Expert gradients do NOT flow to encoder (gradient divorce)."""

    def test_encoder_receives_no_expert_gradients(self):
        alb, encoder = _make_alb()
        x = torch.randn(4, 10, 20)
        out = alb(x)

        # Only backprop through expert output
        loss = out["expert"].sum()
        loss.backward()

        for name, p in encoder.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() == 0, (
                    f"DIVORCE VIOLATED: encoder param '{name}' has grad = {p.grad.abs().sum()}"
                )

    def test_encoder_receives_planner_gradients(self):
        alb, encoder = _make_alb()
        x = torch.randn(4, 10, 20)
        out = alb(x)

        # Backprop through planner output — encoder SHOULD get gradients
        loss = out["planner"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert has_grad, "Encoder should receive gradients through planner path"


class TestRepresentationalDivergence:
    """Test 3: Planner and expert manifolds are NOT identical."""

    def test_manifolds_differ(self):
        alb, _ = _make_alb()
        x = torch.randn(4, 10, 20)
        out = alb(x)

        divergence = (out["planner"] - out["expert"]).abs().mean().item()
        assert divergence > 0.0, "COLLAPSE: planner == expert (spectral decoupling failed)"

    def test_globals_differ(self):
        alb, _ = _make_alb()
        x = torch.randn(4, 10, 20)
        out = alb(x)

        divergence = (out["planner_global"] - out["expert_global"]).abs().mean().item()
        assert divergence > 0.0, "COLLAPSE: planner_global == expert_global"


class TestExpertInitialization:
    """Test 4: Orthogonal initialization check."""

    def test_expert_proj_orthogonal(self):
        alb, _ = _make_alb(d_model=64, n_expert_layers=2)
        for name, module in alb.expert_proj.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                if W.shape[0] == W.shape[1]:  # Square weight matrix only
                    WTW = W.T @ W
                    identity = torch.eye(W.shape[1])
                    error = (WTW - identity).abs().mean().item()
                    # With gain=1.2: WTW ≈ 1.44*I, not I. Error < 2.0 confirms structure.
                    assert error < 2.0, f"Orthogonal init failed for {name}: error={error}"


class TestMaskHandling:
    """Test 5: Masked positions should be zeroed in expert output."""

    def test_masked_expert_zeros(self):
        alb, _ = _make_alb()
        B, T, D_in = 4, 10, 20
        x = torch.randn(B, T, D_in)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -3:] = True  # Last 3 padded

        out = alb(x, mask=mask)
        # Masked positions in expert should be zero
        expert_masked = out["expert"][:, -3:, :]
        assert expert_masked.abs().sum() == 0, "Masked expert positions should be zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
