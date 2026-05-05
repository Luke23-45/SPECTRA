"""
Experiment Templates

Reusable templates for creating new experiments:
    - experiment_template.py: Base structure for new experiments
"""

__all__ = []
"""Experiment templates and shared CLI helpers."""

from spectra.empirical.experiments.templates.experiment_template import add_common_arguments, build_comparison_config

__all__ = ["add_common_arguments", "build_comparison_config"]
