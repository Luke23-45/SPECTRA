"""
Common utilities for the BPGS analysis framework.

Provides shared I/O, metrics, statistics, styling, LaTeX formatting,
configuration loading, and constant definitions used across all
study-specific modules.
"""

from analysis.common.config import (
    get_config,
    get_data_root,
    get_study_data_dir,
    get_output_root,
    get_seeds,
    get_methods,
)
from analysis.common.constants import (
    METHOD_DISPLAY,
    METHOD_ORDER,
    METHOD_COLORS,
    METHOD_MARKERS,
    SEEDS,
    METRIC_DIRECTION,
)
from analysis.common.style import apply_neurips_style, create_figure, save_figure

