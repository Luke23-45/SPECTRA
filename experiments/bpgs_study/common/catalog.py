from __future__ import annotations

from typing import Protocol, runtime_checkable

from experiments.bpgs_study.common.runtime import RunRequest, Stage
from experiments.bpgs_study.common.specs import StudySpec


@runtime_checkable
class StudyRunner(Protocol):
    """Protocol that every study group runner must implement.

    Each study group (ablation, stress, real_data, optional) provides a
    concrete class that owns its execution planning, command building,
    override scoping, and validation.  No core logic lives in common.
    """

    def load_studies(self) -> list[StudySpec]:
        """Return the StudySpec objects for all studies in this group."""
        ...

    def plan(self, study: StudySpec, request: RunRequest) -> list[Stage]:
        """Plan execution stages for *study* given *request*.

        The runner has full control over:
        - which overrides apply to which variants (method-scoping)
        - what command to build for each variant
        - what pre/post stages to include (e.g. subset preparation)
        - study-specific validation
        """
        ...


class StudyEntry:
    """Binds a StudySpec to the runner that owns it."""

    __slots__ = ("spec", "runner")

    def __init__(self, spec: StudySpec, runner: StudyRunner) -> None:
        self.spec = spec
        self.runner = runner

    def plan(self, request: RunRequest) -> list[Stage]:
        return self.runner.plan(self.spec, request)


def _collect_entries() -> list[StudyEntry]:
    from experiments.bpgs_study.ablation.runner import AblationRunner
    from experiments.bpgs_study.optional.runner import OptionalRunner
    from experiments.bpgs_study.real_data.runner import RealDataRunner
    from experiments.bpgs_study.stress.runner import StressRunner

    runners: list[StudyRunner] = [
        AblationRunner(),
        StressRunner(),
        RealDataRunner(),
        OptionalRunner(),
    ]

    entries: list[StudyEntry] = []
    for runner in runners:
        for spec in runner.load_studies():
            entries.append(StudyEntry(spec=spec, runner=runner))

    names = [entry.spec.name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate study registrations detected: {joined}")
    return entries


def _study_sort_key(study: StudySpec) -> tuple[int, str]:
    prefix = study.name.split("_", 1)[0]
    try:
        return (int(prefix), study.name)
    except ValueError:
        return (9999, study.name)


def get_registered_study_entries() -> list[StudyEntry]:
    entries = _collect_entries()
    return sorted(entries, key=lambda entry: _study_sort_key(entry.spec))


def get_registered_studies() -> list[StudySpec]:
    return [entry.spec for entry in get_registered_study_entries()]


def get_study_entry(name: str) -> StudyEntry:
    for entry in get_registered_study_entries():
        if entry.spec.name == name:
            return entry
    raise KeyError(f"Unknown study: {name}")
