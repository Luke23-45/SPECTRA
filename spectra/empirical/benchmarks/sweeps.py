"""Grid sweep definitions for empirical experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Sequence

from spectra.empirical.generators.base import SyntheticGeneratorConfig


PAPER_CONFIG: Dict[str, Sequence[object]] = {
    "task_count": [4, 8],
    "correlation": [-0.5, 0.0, 0.5],
    "imbalance_ratio": [1.0, 10.0],
    "label_noise": [0.0, 0.2],
    "task_type_mix": ["mixed"],
    "seeds": [42, 123, 999],
}

FULL_SWEEP_CONFIG: Dict[str, Sequence[object]] = {
    "task_count": [2, 4, 8, 16],
    "correlation": [-0.8, -0.4, 0.0, 0.4, 0.8],
    "imbalance_ratio": [1.0, 5.0, 20.0, 100.0],
    "label_noise": [0.0, 0.1, 0.3, 0.5],
    "task_type_mix": ["regression", "classification", "mixed"],
    "seeds": [42, 123, 999, 2024, 7777, 31415],
}


@dataclass(frozen=True)
class SweepPoint:
    config: SyntheticGeneratorConfig
    seed: int


def iter_sweep(grid: Dict[str, Sequence[object]], base: SyntheticGeneratorConfig | None = None) -> Iterator[SweepPoint]:
    if "seeds" not in grid:
        raise ValueError("Sweep grid must include 'seeds'.")
    base = base or SyntheticGeneratorConfig()
    keys = [key for key in grid.keys() if key != "seeds"]
    for values in product(*(grid[key] for key in keys)):
        payload = base.to_dict()
        for key, value in zip(keys, values):
            payload[key] = value
        for seed in grid["seeds"]:
            payload["seed"] = int(seed)
            yield SweepPoint(config=SyntheticGeneratorConfig(**payload), seed=int(seed))


def materialize_sweep(grid: Dict[str, Sequence[object]], base: SyntheticGeneratorConfig | None = None) -> List[SweepPoint]:
    return list(iter_sweep(grid, base=base))

