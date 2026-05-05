"""Synthetic family with strongly heterogeneous task types, scales, and noise models."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from spectra.empirical.generators.base import SyntheticGenerator, TaskSpec


class HeterogeneousControlledGenerator(SyntheticGenerator):
    """Generate a challenging mixed-task benchmark with diverse losses and metrics."""

    family_name = "heterogeneous"

    def _generate_core(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[TaskSpec], Dict[str, Any]]:
        x, z, _ = self._sample_representation()
        task_specs: List[TaskSpec] = [
            TaskSpec(name="task_depth_like", task_type="regression", loss="mse"),
            TaskSpec(name="task_robust_reg", task_type="regression", loss="smooth_l1"),
            TaskSpec(name="task_binary_clean", task_type="binary", loss="bce"),
            TaskSpec(name="task_binary_tail", task_type="binary", loss="bce"),
            TaskSpec(name="task_multi10", task_type="multiclass", loss="cross_entropy", output_dim=10, num_classes=10),
            TaskSpec(name="task_multi20", task_type="multiclass", loss="cross_entropy", output_dim=20, num_classes=20),
            TaskSpec(name="task_normals_like", task_type="angular", loss="angular", output_dim=3),
            TaskSpec(name="task_vector_reg", task_type="regression", loss="mae", output_dim=2),
        ]

        anchor = torch.randn(self.config.shared_dim, generator=self._generator)
        directions = self._correlated_vectors(anchor, 24, self.config.correlation)
        targets: Dict[str, torch.Tensor] = {}
        roles: Dict[str, str] = {}

        depth_margin = z @ directions[0]
        hetero_noise = 0.05 + 0.22 * torch.sigmoid(2.5 * (z @ directions[1]))
        targets["task_depth_like"] = (1.4 * depth_margin + self._noise((self.config.n_samples,)) * hetero_noise).float()
        roles["task_depth_like"] = "heteroscedastic_regression"

        robust_margin = z @ directions[2]
        outlier_noise = self._noise((self.config.n_samples,), scale=0.07)
        outlier_mask = (torch.rand(self.config.n_samples, generator=self._generator) < 0.12).float()
        targets["task_robust_reg"] = (
            0.8 * robust_margin + outlier_noise + 2.2 * outlier_mask * self._noise((self.config.n_samples,), scale=0.3)
        ).float()
        roles["task_robust_reg"] = "heavy_tail_regression"

        clean_logits = 1.6 * (z @ directions[3]) - 0.25 * torch.sign(z[:, 0])
        clean_probs = torch.sigmoid(clean_logits / self.config.classification_temperature)
        clean_labels = torch.bernoulli(clean_probs, generator=self._generator)
        targets["task_binary_clean"] = self._flip_binary_labels(clean_labels).float()
        roles["task_binary_clean"] = "balanced_binary"

        tail_logits = 1.1 * (z @ directions[4]) + torch.logit(torch.tensor(0.18))
        tail_probs = torch.sigmoid(tail_logits / max(self.config.classification_temperature, 0.75))
        tail_labels = torch.bernoulli(tail_probs, generator=self._generator)
        targets["task_binary_tail"] = self._flip_binary_labels(tail_labels).float()
        roles["task_binary_tail"] = "imbalanced_binary"

        multi10_vectors = [directions[5]]
        multi10_vectors.extend(self._correlated_vectors(directions[6], 9, max(self.config.correlation, 0.05)))
        logits10 = torch.stack([1.05 * (z @ vec) for vec in multi10_vectors], dim=1)
        logits10 = logits10 + 0.08 * torch.stack([z[:, i % z.shape[1]] for i in range(10)], dim=1)
        labels10 = torch.argmax(logits10, dim=1)
        targets["task_multi10"] = self._flip_multiclass_labels(labels10, 10).long()
        roles["task_multi10"] = "multiclass_10"

        multi20_vectors = [directions[7]]
        multi20_vectors.extend(self._correlated_vectors(-directions[8], 19, 0.0))
        logits20 = torch.stack([0.9 * (z @ vec) for vec in multi20_vectors], dim=1)
        logits20[:, 0] += 0.5
        logits20[:, 1] -= 0.35
        labels20 = torch.argmax(logits20, dim=1)
        targets["task_multi20"] = self._flip_multiclass_labels(labels20, 20).long()
        roles["task_multi20"] = "multiclass_20"

        normals_basis = torch.stack(
            [
                z @ directions[9],
                z @ directions[10],
                z @ directions[11],
            ],
            dim=1,
        )
        normals_target = torch.nn.functional.normalize(normals_basis + 0.12 * self._noise(normals_basis.shape), dim=1)
        targets["task_normals_like"] = normals_target.float()
        roles["task_normals_like"] = "angular_vector"

        vector_target = torch.stack(
            [
                0.75 * (z @ directions[12]) + 0.1 * torch.sign(z[:, 2]),
                -0.6 * (z @ directions[13]) + 0.15 * torch.sin(z[:, 3]),
            ],
            dim=1,
        )
        targets["task_vector_reg"] = vector_target.float()
        roles["task_vector_reg"] = "multioutput_l1_regression"

        metadata = {
            "control_axis": "heterogeneous_regime",
            "task_roles": roles,
            "num_tasks": len(task_specs),
            "description": "Mixed regression, binary, multiclass, angular, and multi-output regression tasks.",
        }
        return x.float(), targets, task_specs, metadata
