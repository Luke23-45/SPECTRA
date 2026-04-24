"""Deep-dive simulation sweep for BPGS math variants.

This script evaluates uncertainty-objective variants in a NYUv2-like three-task
setting with conflicting gradients and heterogeneous loss scales.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch


@dataclass
class Variant:
    name: str
    normalization: str
    linear_coef: float


def _normalize(losses: torch.Tensor, mode: str, floor: float = 1e-8) -> torch.Tensor:
    safe = losses.clamp_min(floor)
    if mode == "raw":
        return safe
    if mode == "mean":
        return safe / safe.mean().clamp_min(floor)
    if mode == "max":
        return safe / safe.max().clamp_min(floor)
    if mode == "geo":
        return safe / torch.exp(torch.mean(torch.log(safe))).clamp_min(floor)
    raise ValueError(f"Unknown normalization mode: {mode}")


def run_trial(variant: Variant, seed: int, steps: int = 600) -> dict:
    torch.manual_seed(seed)

    # Shared parameter vector with task optima deliberately misaligned
    x = torch.nn.Parameter(torch.zeros(2))
    theta = torch.nn.Parameter(torch.zeros(3))

    # NYUv2-like heterogeneous scales: seg, depth, normals
    scales = torch.tensor([1.3, 0.25, 0.9])
    mus = torch.tensor([[1.2, -0.5], [-0.7, 1.0], [0.6, 1.3]])
    As = torch.stack(
        [
            torch.tensor([[2.0, 0.35], [0.35, 1.4]]),
            torch.tensor([[1.1, -0.2], [-0.2, 1.6]]),
            torch.tensor([[1.8, 0.25], [0.25, 1.1]]),
        ]
    )

    opt_x = torch.optim.Adam([x], lr=0.02)
    opt_theta = torch.optim.Adam([theta], lr=0.04)

    for _ in range(steps):
        raw_losses = []
        for i in range(3):
            d = x - mus[i]
            quad = 0.5 * d @ As[i] @ d
            raw_losses.append(scales[i] * quad + 0.01)
        raw = torch.stack(raw_losses)
        # Unit drift imitates heterogeneous metric scales seen in real MTL pipelines.
        drift = torch.exp(0.5 * torch.randn(3))
        observed = raw * drift

        s = -8.0 + 16.0 * torch.sigmoid(theta)
        precision = torch.exp(-s)

        # Network update
        opt_x.zero_grad()
        j_net = 0.5 * (precision.detach() * observed).sum()
        j_net.backward()
        opt_x.step()

        # Uncertainty update
        opt_theta.zero_grad()
        detached = _normalize(observed.detach(), variant.normalization)
        j_unc = 0.5 * (precision * detached).sum() + variant.linear_coef * s.sum()
        j_unc.backward()
        opt_theta.step()

    with torch.no_grad():
        final_raw = []
        for i in range(3):
            d = x - mus[i]
            quad = 0.5 * d @ As[i] @ d
            final_raw.append((scales[i] * quad + 0.01).item())
        final_raw = torch.tensor(final_raw)
        # lower score is better
        score = float(final_raw.mean() + 0.2 * final_raw.std())

    return {
        "variant": variant.name,
        "normalization": variant.normalization,
        "linear_coef": variant.linear_coef,
        "seed": seed,
        "seg_loss": float(final_raw[0]),
        "depth_loss": float(final_raw[1]),
        "normals_loss": float(final_raw[2]),
        "score": score,
    }


def main() -> None:
    out_dir = Path("temp/bpgs_math_lab/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    for norm, coef in product(["raw", "mean", "max", "geo"], [0.35, 0.5, 0.65]):
        variants.append(Variant(name=f"{norm}_c{coef}", normalization=norm, linear_coef=coef))

    rows = []
    for v in variants:
        for seed in range(7):
            rows.append(run_trial(v, seed=seed))

    csv_path = out_dir / "variant_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for v in variants:
        v_rows = [r for r in rows if r["variant"] == v.name]
        summary[v.name] = {
            "normalization": v.normalization,
            "linear_coef": v.linear_coef,
            "mean_score": sum(r["score"] for r in v_rows) / len(v_rows),
            "mean_seg_loss": sum(r["seg_loss"] for r in v_rows) / len(v_rows),
            "mean_depth_loss": sum(r["depth_loss"] for r in v_rows) / len(v_rows),
            "mean_normals_loss": sum(r["normals_loss"] for r in v_rows) / len(v_rows),
        }

    ranking = sorted(summary.items(), key=lambda kv: kv[1]["mean_score"])
    report = {
        "best": {"name": ranking[0][0], **ranking[0][1]},
        "runner_up": {"name": ranking[1][0], **ranking[1][1]},
        "worst": {"name": ranking[-1][0], **ranking[-1][1]},
        "ranking": [{"name": name, **vals} for name, vals in ranking],
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Best variant: {report['best']['name']} (score={report['best']['mean_score']:.6f})")


if __name__ == "__main__":
    main()
