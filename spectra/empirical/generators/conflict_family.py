"""Synthetic family with explicit gradient and decision conflict controls."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from spectra.empirical.generators.base import TaskSpec, SyntheticGenerator


class ConflictControlledGenerator(SyntheticGenerator):
    """Generate tasks with aligned, orthogonal, and opposing label functions."""

    family_name = "conflict"

    def _generate_core(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[TaskSpec], Dict[str, Any]]:
        x, z, _ = self._sample_representation()
        task_specs = self._make_task_specs()
        base = torch.randn(self.config.shared_dim, generator=self._generator)
        base = base / base.norm().clamp(min=1e-6)
        conflict_strength = float(max(0.0, -self.config.correlation))
        targets: Dict[str, torch.Tensor] = {}
        descriptors: Dict[str, str] = {}

        for index, spec in enumerate(task_specs):
            if index % 3 == 0:
                direction = base
                descriptors[spec.name] = "aligned"
            elif index % 3 == 1:
                direction = -base * max(0.6, conflict_strength)
                descriptors[spec.name] = "opposed"
            else:
                noise = torch.randn(base.shape, generator=self._generator, dtype=base.dtype, device=base.device)
                noise = noise - torch.dot(noise, base) * base
                direction = noise / noise.norm().clamp(min=1e-6)
                descriptors[spec.name] = "orthogonal"

            margin = z @ direction
            if spec.task_type == "regression":
                target = margin + 0.2 * torch.sign(z[:, 0]) * (index + 1)
                target = target + self._noise((self.config.n_samples,), scale=0.1 + self.config.label_noise)
                targets[spec.name] = target.float()
            elif spec.task_type == "binary":
                logits = 2.0 * margin + 0.75 * torch.sign(z[:, 1 % z.shape[1]])
                probs = torch.sigmoid(logits)
                labels = torch.bernoulli(probs, generator=self._generator)
                targets[spec.name] = self._flip_binary_labels(labels).float()
            else:
                num_classes = spec.num_classes or 3
                class_vectors = [direction]
                class_vectors.extend(self._correlated_vectors(-direction, num_classes - 1, 0.0))
                logits = torch.stack([z @ vec for vec in class_vectors], dim=1)
                if descriptors[spec.name] == "opposed":
                    logits = torch.flip(logits, dims=(1,))
                labels = torch.argmax(logits, dim=1)
                targets[spec.name] = self._flip_multiclass_labels(labels, num_classes).long()

        metadata = {
            "control_axis": "conflict",
            "requested_conflict_level": conflict_strength,
            "task_roles": descriptors,
        }
        return x.float(), targets, task_specs, metadata
