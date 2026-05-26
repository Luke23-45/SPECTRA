from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectra.data.nyuv2.dataset import IGNORE_INDEX, NUM_CLASSES, NYUv2Dataset, resolve_nyuv2_root
from experiments.bpgs_study.definitions import SUBSET_ROOT, nyuv2_subset_id


SUBSET_VERSION = "v1"
BUDGETS = {"25": 199, "50": 398, "75": 596}
SUBSET_SEEDS = [11, 22, 33]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_remainder(total: int, buckets: Dict[str, int]) -> Dict[str, int]:
    bucket_total = sum(buckets.values())
    if bucket_total <= 0:
        return {key: 0 for key in buckets}
    exact = {key: (count / bucket_total) * total for key, count in buckets.items()}
    base = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(base.values())
    order = sorted(((key, exact[key] - base[key]) for key in buckets), key=lambda item: (-item[1], item[0]))
    for key, _ in order[:remainder]:
        base[key] += 1
    return base


def _sample_stratified(entries_by_stratum: Dict[str, List[int]], target_size: int, rng: Random) -> List[int]:
    allocations = _largest_remainder(target_size, {k: len(v) for k, v in entries_by_stratum.items()})
    selected: List[int] = []
    leftovers: List[int] = []
    for stratum, entries in sorted(entries_by_stratum.items()):
        if allocations[stratum] >= len(entries):
            selected.extend(entries)
            continue
        chosen = set(rng.sample(entries, allocations[stratum]))
        selected.extend(sorted(chosen))
        leftovers.extend(idx for idx in entries if idx not in chosen)
    if len(selected) < target_size:
        selected.extend(rng.sample(leftovers, target_size - len(selected)))
    return sorted(selected[:target_size])


def _majority_class(label: torch.Tensor) -> int:
    valid = label[(label >= 0) & (label < NUM_CLASSES)]
    if valid.numel() == 0:
        return IGNORE_INDEX
    counts = torch.bincount(valid.view(-1), minlength=NUM_CLASSES)
    return int(torch.argmax(counts).item())


def _depth_bucket(valid_ratio: float) -> str:
    if valid_ratio < 0.25:
        return "depth_q0"
    if valid_ratio < 0.50:
        return "depth_q1"
    if valid_ratio < 0.75:
        return "depth_q2"
    return "depth_q3"


def _subset_summary(descriptors: Dict[int, Dict[str, object]], indices: Iterable[int]) -> Dict[str, object]:
    selected = list(indices)
    class_presence = Counter()
    valid_depth_values: List[float] = []
    for idx in selected:
        desc = descriptors[idx]
        for cls in desc["present_classes"]:
            class_presence[str(cls)] += 1
        valid_depth_values.append(float(desc["valid_depth_ratio"]))
    return {
        "image_count": len(selected),
        "class_image_counts": dict(sorted(class_presence.items(), key=lambda item: int(item[0]))),
        "mean_valid_depth_ratio": sum(valid_depth_values) / max(len(valid_depth_values), 1),
        "min_valid_depth_ratio": min(valid_depth_values) if valid_depth_values else 0.0,
        "max_valid_depth_ratio": max(valid_depth_values) if valid_depth_values else 0.0,
    }


def build_subsets(root: str, output_dir: Path) -> List[Path]:
    dataset = NYUv2Dataset(root=root, split="train", augmentation=False, normalize_rgb=False, subset_pct=1.0)
    manifest_path = dataset.root / "train_index.json"
    source_hash = _sha256(manifest_path)

    descriptors: Dict[int, Dict[str, object]] = {}
    strata: Dict[str, List[int]] = defaultdict(list)

    for idx, sample_meta in enumerate(dataset.samples):
        _, label_bytes, depth_bytes, _ = dataset._read_sample_bytes(sample_meta)
        hw = sample_meta["shape_hw"]
        label = dataset._decode_uint8_label(label_bytes, hw).to(dtype=torch.long)
        depth = dataset._decode_float16_map(depth_bytes, hw, channels=1, layout=dataset.depth_layout)
        valid_depth_ratio = float((depth > 0.0).float().mean().item())
        present_classes = sorted(int(v) for v in torch.unique(label[(label >= 0) & (label < NUM_CLASSES)]).tolist())
        majority = _majority_class(label)
        stratum = f"class_{majority}|{_depth_bucket(valid_depth_ratio)}"

        descriptors[idx] = {
            "present_classes": present_classes,
            "valid_depth_ratio": valid_depth_ratio,
            "stratum": stratum,
        }
        strata[stratum].append(idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    full_summary = _subset_summary(descriptors, range(len(dataset.samples)))
    manifest_records = []

    for budget, count in BUDGETS.items():
        for subset_seed in SUBSET_SEEDS:
            subset_id = nyuv2_subset_id(budget, subset_seed)
            indices = _sample_stratified(strata, count, Random(subset_seed))
            payload = {
                "subset_id": subset_id,
                "indices": indices,
                "metadata": {
                    "version": SUBSET_VERSION,
                    "dataset_name": "nyuv2",
                    "source_manifest": str(manifest_path),
                    "source_manifest_sha256": source_hash,
                    "train_size": len(dataset.samples),
                    "eval_size": 654,
                    "subset_budget_pct": int(budget),
                    "subset_size": count,
                    "subset_seed": subset_seed,
                    "sampling_mode": "image_stratified",
                    "full_train_summary": full_summary,
                    "subset_summary": _subset_summary(descriptors, indices),
                },
            }
            out_path = output_dir / f"{subset_id}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written.append(out_path)
            manifest_records.append({"subset_id": subset_id, "path": str(out_path), "subset_size": count, "subset_seed": subset_seed})

    manifest = {
        "version": SUBSET_VERSION,
        "dataset_name": "nyuv2",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": source_hash,
        "subsets": manifest_records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate isolated NYUv2 subsets for the BPGS objective study.")
    parser.add_argument("--root", default="datasets/nyuv2_lmdb/data")
    parser.add_argument("--output-dir", type=Path, default=SUBSET_ROOT)
    args = parser.parse_args()

    written = build_subsets(str(resolve_nyuv2_root(args.root)), args.output_dir)
    print(f"Wrote {len(written)} subset files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
