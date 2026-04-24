import torch

from spectra.core.bpgs import BPGS


def _step_theta_once(bpgs: BPGS, raw_losses):
    opt = torch.optim.SGD(bpgs.parameters(), lr=0.1)
    opt.zero_grad()
    loss_unc = bpgs.uncertainty_loss(raw_losses)
    loss_unc.backward()
    grad = bpgs.theta.grad.detach().clone()
    opt.step()
    return grad


def test_relative_loss_invariance_matches_under_global_rescaling():
    base = [torch.tensor(1.2), torch.tensor(0.3), torch.tensor(0.8)]
    scaled = [10.0 * x for x in base]

    bpgs_a = BPGS(num_tasks=3, relative_loss_invariance=True, relative_loss_mode="max")
    bpgs_b = BPGS(num_tasks=3, relative_loss_invariance=True, relative_loss_mode="max")

    grad_a = _step_theta_once(bpgs_a, base)
    grad_b = _step_theta_once(bpgs_b, scaled)

    torch.testing.assert_close(grad_a, grad_b, rtol=1e-4, atol=1e-4)


def test_raw_mode_changes_under_global_rescaling():
    base = [torch.tensor(1.2), torch.tensor(0.3), torch.tensor(0.8)]
    scaled = [10.0 * x for x in base]

    bpgs_a = BPGS(num_tasks=3, relative_loss_invariance=False)
    bpgs_b = BPGS(num_tasks=3, relative_loss_invariance=False)

    grad_a = _step_theta_once(bpgs_a, base)
    grad_b = _step_theta_once(bpgs_b, scaled)

    assert not torch.allclose(grad_a, grad_b, rtol=1e-4, atol=1e-4)
