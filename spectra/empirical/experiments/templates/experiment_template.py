"""Shared command-line scaffold for empirical experiments."""

from __future__ import annotations

import argparse
from typing import Sequence

from spectra.empirical.runners.method_comparison import ComparisonConfig, MethodComparisonHarness


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--methods", nargs="+", default=["bpgs", "uwso", "static"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/empirical")
    return parser


def build_comparison_config(args: argparse.Namespace, family: str) -> ComparisonConfig:
    return ComparisonConfig(
        family=family,
        methods=args.methods,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.output_dir,
    )
