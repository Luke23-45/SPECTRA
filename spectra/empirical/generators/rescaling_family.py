"""Synthetic family for pure post-hoc loss-rescaling stress tests."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from spectra.empirical.generators.base import SyntheticGenerator, TaskSpec


class RescalingControlledGenerator(SyntheticGenerator):
    """Generate a fixed heterogeneous task bundle; stress is injected only in training losses."""

    family_name = "rescaling"

    def _generate_core(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[TaskSpec], Dict[str, Any]]:
        x, z, _ = self._sample_representation()

        task_specs: List[TaskSpec] = [
            TaskSpec(name="task_reg_mse", task_type="regression", loss="mse"),
            TaskSpec(name="task_reg_smoothl1", task_type="regression", loss="smooth_l1"),
            TaskSpec(name="task_binary", task_type="binary", loss="bce"),
            TaskSpec(name="task_multi6", task_type="multiclass", loss="cross_entropy", output_dim=6, num_classes=6),
            TaskSpec(name="task_angular", task_type="angular", loss="angular", output_dim=3),
        ]

        anchor = torch.randn(self.config.shared_dim, generator=self._generator)
        directions = self._correlated_vectors(anchor, len(task_specs) + 4, max(self.config.correlation, 0.25))
        targets: Dict[str, torch.Tensor] = {}

        reg_margin = z @ directions[0]
        targets["task_reg_mse"] = (
            1.25 * reg_margin + self._noise((self.config.n_samples,), scale=0.12 + self.config.label_noise)
        ).float()

        smooth_margin = z @ directions[1]
        outlier_noise = self._noise((self.config.n_samples,), scale=0.08)
        outlier_mask = (torch.rand(self.config.n_samples, generator=self._generator) < 0.08).float()
        targets["task_reg_smoothl1"] = (
            0.9 * smooth_margin + outlier_noise + 1.75 * outlier_mask * self._noise((self.config.n_samples,), scale=0.18)
        ).float()

        binary_margin = z @ directions[2] + 0.35 * torch.sign(z[:, 0])
        binary_probs = torch.sigmoid(binary_margin / self.config.classification_temperature)
        binary_labels = torch.bernoulli(binary_probs, generator=self._generator)
        targets["task_binary"] = self._flip_binary_labels(binary_labels).float()

        multiclass_vectors = [directions[3]]
        multiclass_vectors.extend(self._correlated_vectors(directions[4], 5, 0.15))
        multiclass_logits = torch.stack([1.2 * (z @ vec) for vec in multiclass_vectors], dim=1)
        multiclass_logits = multiclass_logits + 0.15 * torch.stack([z[:, i % z.shape[1]] for i in range(6)], dim=1)
        multiclass_labels = torch.argmax(multiclass_logits, dim=1)
        targets["task_multi6"] = self._flip_multiclass_labels(multiclass_labels, 6).long()

        angular_basis = torch.stack(
            [
                z @ directions[5],
                z @ directions[6],
                z @ directions[7],
            ],
            dim=1,
        )
        angular_target = torch.nn.functional.normalize(angular_basis + 0.08 * self._noise(angular_basis.shape), dim=1)
        targets["task_angular"] = angular_target.float()

        metadata = {
            "control_axis": "loss_rescaling",
            "task_bundle": [spec.name for spec in task_specs],
            "description": "Underlying targets are fixed; only one task loss is rescaled during training.",
        }
        return x.float(), targets, task_specs, metadata
