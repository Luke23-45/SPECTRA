"""Generate publication-oriented summary tables and plots from empirical outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from spectra.empirical.analysis.comparative_analysis import summarize_comparison_frame


def collect_results(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in results_dir.rglob("result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"method": payload["method"], "seed": payload["seed"], "path": str(path)}
        row.update(payload["final_metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(results_dir: Path, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_results(results_dir)
    if frame.empty:
        raise ValueError(f"No result.json files found under {results_dir}.")
    summary = summarize_comparison_frame(frame)
    csv_path = output_dir / "summary.csv"
    tex_path = output_dir / "summary.tex"
    md_path = output_dir / "summary.md"
    frame.to_csv(output_dir / "all_results.csv", index=False)
    summary.to_csv(csv_path, index=False)
    summary.to_latex(tex_path, index=False, float_format=lambda value: f"{value:.4f}")
    md_path.write_text(summary.to_markdown(index=False), encoding="utf-8")
    return {"summary_csv": str(csv_path), "summary_tex": str(tex_path), "summary_md": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate empirical experiment summary tables.")
    parser.add_argument("--results-dir", default="outputs/empirical", type=Path)
    parser.add_argument("--output-dir", default="outputs/empirical_report", type=Path)
    args = parser.parse_args()
    paths = write_report(args.results_dir, args.output_dir)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()

