from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from omegaconf import OmegaConf

from experiments.bpgs_study.common.specs import EmpiricalVariant, StudySpec, TrainingVariant


def _as_list(value: Iterable[Any] | None) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _load_raw_config(path: Path) -> Dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Study config must resolve to a mapping: {path}")
    return dict(data)


def load_study_spec(path: Path) -> StudySpec:
    raw = _load_raw_config(path)
    variants_raw = _as_list(raw.get("variants"))
    kind = str(raw["kind"])
    group = str(raw.get("group", path.parent.parent.name))

    if kind == "training":
        variants = [
            TrainingVariant(
                label=str(item["label"]),
                dataset=str(item["dataset"]),
                method=str(item["method"]),
                epochs=int(item["epochs"]),
                seeds=tuple(int(seed) for seed in _as_list(item["seeds"])),
                extra_overrides=tuple(str(v) for v in _as_list(item.get("extra_overrides"))),
                subset_budget=str(item["subset_budget"]) if item.get("subset_budget") is not None else None,
                subset_seed=int(item["subset_seed"]) if item.get("subset_seed") is not None else None,
                use_subset_file=bool(item.get("use_subset_file", False)),
                early_stop=bool(item.get("early_stop", True)),
                early_stop_patience=int(item["early_stop_patience"]) if item.get("early_stop_patience") is not None else None,
                early_stop_min_delta=float(item["early_stop_min_delta"]) if item.get("early_stop_min_delta") is not None else None,
            )
            for item in variants_raw
        ]
    elif kind == "empirical":
        variants = [
            EmpiricalVariant(
                label=str(item["label"]),
                experiment=str(item["experiment"]),
                family=str(item["family"]),
                methods=tuple(str(method) for method in _as_list(item["methods"])),
                seeds=tuple(int(seed) for seed in _as_list(item["seeds"])),
                overrides=tuple(str(v) for v in _as_list(item.get("overrides"))),
            )
            for item in variants_raw
        ]
    else:
        raise ValueError(f"Unknown study kind '{kind}' in {path}")

    return StudySpec(
        name=str(raw["name"]),
        kind=kind,
        group=group,
        description=str(raw["description"]),
        variants=tuple(variants),
        config_path=path,
        requires_nyuv2_subsets=bool(raw.get("requires_nyuv2_subsets", False)),
        notes=str(raw["notes"]) if raw.get("notes") else None,
    )
