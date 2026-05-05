"""Synthetic family with explicit task correlation control."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from spectra.empirical.generators.base import TaskSpec, SyntheticGenerator, SyntheticGeneratorConfig


class CorrelationControlledGenerator(SyntheticGenerator):
    """Generate tasks around a shared anchor with tunable pairwise alignment."""

    family_name = "correlation"

    def __init__(self, config: SyntheticGeneratorConfig) -> None:
        super().__init__(config)

    def _generate_core(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[TaskSpec], Dict[str, Any]]:
        x, z, _ = self._sample_representation()
        task_specs = self._make_task_specs()
        anchor = torch.randn(self.config.shared_dim, generator=self._generator)
        vectors = self._correlated_vectors(anchor, len(task_specs), self.config.correlation)
        targets: Dict[str, torch.Tensor] = {}
        scales: Dict[str, float] = {}

        for index, (spec, direction) in enumerate(zip(task_specs, vectors)):
            scale = 1.0 + 0.1 * index
            scales[spec.name] = scale
            margin = z @ direction
            if spec.task_type == "regression":
                target = scale * margin + self.config.regression_bias_scale * self._noise((self.config.n_samples,))
                targets[spec.name] = target.float()
            elif spec.task_type == "binary":
                logits = margin / self.config.classification_temperature
                probs = torch.sigmoid(logits)
                labels = torch.bernoulli(probs, generator=self._generator)
                targets[spec.name] = self._flip_binary_labels(labels).float()
            else:
                class_vectors = self._correlated_vectors(direction, spec.num_classes or 3, self.config.correlation)
                logits = torch.stack([z @ vec for vec in class_vectors], dim=1)
                labels = torch.argmax(logits, dim=1)
                targets[spec.name] = self._flip_multiclass_labels(labels, spec.num_classes or 3).long()

        metadata = {
            "control_axis": "correlation",
            "requested_correlation": self.config.correlation,
            "task_scales": scales,
        }
        return x.float(), targets, task_specs, metadata

