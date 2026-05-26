from __future__ import annotations

from experiments.bpgs_study.common.runtime import RunRequest
from experiments.bpgs_study.common.specs import OverrideSpec


def validate_run_request(request: RunRequest) -> None:
    """Validate a RunRequest, ensuring all overrides are well-formed.

    Since overrides are now parsed into OverrideSpec objects at construction
    time, this primarily checks for logical consistency rather than syntax.
    """
    for override in request.overrides:
        if not override.key:
            raise ValueError(
                f"Invalid override '{override.raw}': key must be non-empty."
            )
        if override.scope_method is not None and not override.scope_method:
            raise ValueError(
                f"Invalid override '{override.raw}': scope method must be non-empty "
                f"when using 'scope:key=value' form."
            )


def parse_raw_overrides(raw_overrides: tuple[str, ...]) -> tuple[OverrideSpec, ...]:
    """Parse raw ``key=value`` / ``method:key=value`` strings into OverrideSpec objects."""
    return tuple(OverrideSpec.parse(raw) for raw in raw_overrides)
