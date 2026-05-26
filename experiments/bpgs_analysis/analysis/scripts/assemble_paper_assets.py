"""
Assemble the final paper-ready figure and table bundle.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)


MAIN_FIGURES = [
    ("ablation_per_task_val_curves.pdf", "Fig1_ablation_per_task_val_curves.pdf"),
    ("ablation_weight_dynamics.pdf", "Fig2_ablation_weight_dynamics.pdf"),
    ("nyuv2_radar.pdf", "Fig3_nyuv2_radar.pdf"),
    ("full_data_task_heatmaps.pdf", "Fig4_full_data_task_heatmaps.pdf"),
    ("stress_robustness_summary.pdf", "Fig5_stress_robustness_summary.pdf"),
    ("degradation_bar.pdf", "Fig6_degradation_bar.pdf"),
]

SUPP_FIGURES = [
    ("ablation_training_curves.pdf", "FigS1_ablation_training_curves.pdf"),
    ("nyuv2_training_curves.pdf", "FigS2_nyuv2_training_curves.pdf"),
    ("yeast_f1_bar.pdf", "FigS3_yeast_f1_bar.pdf"),
    ("rf1_metrics_bar.pdf", "FigS4_rf1_metrics_bar.pdf"),
    ("heterogeneous_grouped_bar.pdf", "FigS5_heterogeneous_grouped_bar.pdf"),
    ("yeast_per_label_heatmap.pdf", "FigS6_yeast_per_label_heatmap.pdf"),
    ("rf1_per_site_heatmap.pdf", "FigS7_rf1_per_site_heatmap.pdf"),
]

TABLES = [
    "ablation_table.tex",
    "nyuv2_main_table.tex",
    "yeast_table.tex",
    "rf1_table.tex",
    "scale_stress_scale_table.tex",
    "scale_stress_rescaling_table.tex",
    "degradation_table.tex",
    "heterogeneous_table.tex",
]


def _copy_assets(src_dir: Path, dst_dir: Path, mapping: list[tuple[str, str]]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in mapping:
        src = src_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing asset: {src}")
        shutil.copy2(src, dst_dir / dst_name)


def _copy_tables(src_dir: Path, dst_dir: Path, filenames: list[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing table: {src}")
        shutil.copy2(src, dst_dir / name)


def _write_manifest(out_dir: Path) -> None:
    manifest = out_dir / "README.md"
    lines = [
        "# Final Paper Assets",
        "",
        "## Main Figures",
        "",
        "1. `Fig1_ablation_per_task_val_curves.pdf`",
        "2. `Fig2_ablation_weight_dynamics.pdf`",
        "3. `Fig3_nyuv2_radar.pdf`",
        "4. `Fig4_full_data_task_heatmaps.pdf`",
        "5. `Fig5_stress_robustness_summary.pdf`",
        "6. `Fig6_degradation_bar.pdf`",
        "",
        "## Supplementary Figures",
        "",
        "1. `FigS1_ablation_training_curves.pdf`",
        "2. `FigS2_nyuv2_training_curves.pdf`",
        "3. `FigS3_yeast_f1_bar.pdf`",
        "4. `FigS4_rf1_metrics_bar.pdf`",
        "5. `FigS5_heterogeneous_grouped_bar.pdf`",
        "6. `FigS6_yeast_per_label_heatmap.pdf`",
        "7. `FigS7_rf1_per_site_heatmap.pdf`",
        "",
        "## Tables",
        "",
    ]
    lines.extend([f"- `{name}`" for name in TABLES])
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    figures_dir = base_dir / "figures" / "pdf"
    tables_dir = base_dir / "tables"
    out_dir = base_dir / "paper_final"

    _copy_assets(figures_dir, out_dir / "main_figures", MAIN_FIGURES)
    _copy_assets(figures_dir, out_dir / "supp_figures", SUPP_FIGURES)
    _copy_tables(tables_dir, out_dir / "tables", TABLES)
    _write_manifest(out_dir)

    logger.info("Assembled final paper assets: %s", out_dir)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Assemble final paper-ready assets.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="analysis/outputs",
        help="Base analysis output directory containing figures/ and tables/.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main(args)


if __name__ == "__main__":
    cli()
