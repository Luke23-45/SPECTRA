"""Synthetic family with explicit task scale and label imbalance controls."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from spectra.empirical.generators.base import TaskSpec, SyntheticGenerator


class ImbalanceControlledGenerator(SyntheticGenerator):
    """Generate tasks whose scales and class priors diverge by a controlled ratio."""

    family_name = "imbalance"

    def _generate_core(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[TaskSpec], Dict[str, Any]]:
        x, z, _ = self._sample_representation()
        task_specs = self._make_task_specs()
        anchor = torch.randn(self.config.shared_dim, generator=self._generator)
        vectors = self._correlated_vectors(anchor, len(task_specs), max(self.config.correlation, 0.0))
        targets: Dict[str, torch.Tensor] = {}

        log_ratio = torch.linspace(0.0, 1.0, len(task_specs))
        scales = torch.pow(torch.tensor(float(self.config.imbalance_ratio)), log_ratio)
        priors = torch.linspace(0.5, 0.5 / self.config.imbalance_ratio, len(task_specs))

        for index, (spec, direction) in enumerate(zip(task_specs, vectors)):
            margin = z @ direction
            scale = scales[index].item()
            if spec.task_type == "regression":
                noise_scale = 0.05 * scale + self.config.label_noise * scale
                target = scale * margin + self._noise((self.config.n_samples,), scale=noise_scale)
                targets[spec.name] = target.float()
            elif spec.task_type == "binary":
                bias = torch.logit(priors[index].clamp(0.02, 0.98))
                logits = (margin + bias) / self.config.classification_temperature
                probs = torch.sigmoid(logits)
                labels = torch.bernoulli(probs, generator=self._generator)
                targets[spec.name] = self._flip_binary_labels(labels).float()
            else:
                num_classes = spec.num_classes or 3
                class_vectors = self._correlated_vectors(direction, num_classes, 0.15)
                logits = torch.stack([scale * (z @ vec) for vec in class_vectors], dim=1)
                logits[:, 0] += torch.log(torch.tensor(float(self.config.imbalance_ratio)))
                labels = torch.argmax(logits, dim=1)
                targets[spec.name] = self._flip_multiclass_labels(labels, num_classes).long()

        metadata = {
            "control_axis": "imbalance",
            "requested_imbalance_ratio": self.config.imbalance_ratio,
            "task_scales": {spec.name: float(scales[index].item()) for index, spec in enumerate(task_specs)},
            "class_priors": {spec.name: float(priors[index].item()) for index, spec in enumerate(task_specs)},
        }
        return x.float(), targets, task_specs, metadata
