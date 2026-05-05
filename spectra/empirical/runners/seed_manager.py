"""Utilities for reproducible paired-seed empirical comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from spectra.utils.seed import seed_everything


DEFAULT_PAIRED_SEEDS: tuple[int, ...] = (42, 123, 999)


@dataclass(frozen=True)
class SeedBundle:
    train_seed: int
    data_seed: int
    eval_seed: int


def build_seed_bundle(seed: int) -> SeedBundle:
    return SeedBundle(train_seed=seed, data_seed=seed + 17, eval_seed=seed + 29)


def paired_seeds(seeds: Sequence[int] | None = None) -> List[SeedBundle]:
    base = seeds or DEFAULT_PAIRED_SEEDS
    return [build_seed_bundle(int(seed)) for seed in base]


def apply_seed_bundle(bundle: SeedBundle, deterministic: bool = True) -> None:
    seed_everything(bundle.train_seed, deterministic=deterministic)

