from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class OverrideSpec:
    """A runtime override with optional method scoping.

    Formats accepted by :meth:`parse`:

    - ``key=value``            → applies to **all** variants
    - ``method_name:key=value`` → applies **only** to variants whose
      ``method`` field matches *method_name*

    The raw string is preserved so error messages always show what the
    user typed.
    """

    raw: str
    key: str
    value: str
    scope_method: str | None = None

    @classmethod
    def parse(cls, raw: str) -> OverrideSpec:
        """Parse a ``key=value`` or ``method:key=value`` string."""
        if "=" not in raw:
            raise ValueError(
                f"Invalid override '{raw}': expected key=value syntax."
            )
        lhs, value = raw.split("=", 1)
        if ":" in lhs:
            scope_method, key = lhs.split(":", 1)
            if not scope_method or not key:
                raise ValueError(
                    f"Invalid override '{raw}': both scope and key must be "
                    f"non-empty in 'scope:key=value' form."
                )
            return cls(raw=raw, key=key, value=value, scope_method=scope_method)
        if not lhs:
            raise ValueError(
                f"Invalid override '{raw}': key must be non-empty."
            )
        return cls(raw=raw, key=lhs, value=value, scope_method=None)

    def applies_to_method(self, method: str) -> bool:
        """Return True if this override should apply to the given method."""
        if self.scope_method is None:
            return True
        return self.scope_method == method

    def to_hydra_arg(self) -> str:
        """Render as a Hydra-compatible ``key=value`` string."""
        return f"{self.key}={self.value}"


@dataclass(frozen=True)
class TrainingVariant:
    label: str
    dataset: str
    method: str
    epochs: int
    seeds: Sequence[int]
    extra_overrides: Sequence[str] = field(default_factory=tuple)
    subset_budget: str | None = None
    subset_seed: int | None = None
    use_subset_file: bool = False
    early_stop: bool = True
    early_stop_patience: int | None = None
    early_stop_min_delta: float | None = None


@dataclass(frozen=True)
class EmpiricalVariant:
    label: str
    experiment: str
    family: str
    methods: Sequence[str]
    seeds: Sequence[int]
    overrides: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class StudySpec:
    name: str
    kind: str
    group: str
    description: str
    variants: Sequence[TrainingVariant | EmpiricalVariant]
    config_path: Path
    requires_nyuv2_subsets: bool = False
    notes: str | None = None
