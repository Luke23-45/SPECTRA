"""
Publication-grade matplotlib style system for NeurIPS 2026.

Design principles
-----------------
* **Premium feel**: generous whitespace, refined typography, subtle grid.
* **Colorblind-safe**: palette verified with Coblis & Viz Palette tools.
* **NeurIPS-compliant**: Type-42 fonts, serif family (Times), correct sizes.
* **Consistent**: every figure uses the same design tokens.

Usage::

    from analysis.common.style import apply_neurips_style, create_figure, save_figure

    with apply_neurips_style():
        fig, ax = create_figure(width="single")
        ...
        save_figure(fig, Path("output/my_figure"))
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# Column widths (inches) for NeurIPS two-column format
_COL_WIDTH   = 3.25    # single column
_TEXT_WIDTH  = 6.875    # full text width (double column)

# DPI for raster outputs
_DPI = 300

# Font sizes (pt) — NeurIPS caption is 9pt, body 10pt
_FONT_SIZE_TITLE  = 10
_FONT_SIZE_LABEL  = 9
_FONT_SIZE_TICK   = 7.5
_FONT_SIZE_LEGEND = 7
_FONT_SIZE_ANNOT  = 6.5


# Colorblind-safe palette (Tol + Wong, verified with Coblis simulator)
PALETTE = {
    # Baselines — cool, muted tones
    "static":               "#88CCEE",   # sky blue
    "kendall":              "#332288",   # indigo
    "uwso":                 "#117733",   # forest green
    "pcgrad":               "#882255",   # wine
    "gradnorm_proxy":       "#CC6677",   # rose

    # BPGS family — warm, prominent tones
    "bpgs":                 "#EE3377",   # magenta-pink (hero color)
    "bpgs_canonical":       "#EE3377",   # same hero
    "bpgs_batch_aware_fixed": "#EE7733", # tangerine
    "bpgs_stateless_auto":  "#CCBB44",   # olive gold
    "bpgs_stateless_fixed": "#44AA99",   # teal
}

# Marker styles — distinct, not too large
MARKERS = {
    "static":               "s",    # square
    "kendall":              "^",    # triangle up
    "uwso":                 "D",    # diamond
    "pcgrad":               "v",    # triangle down
    "gradnorm_proxy":       "P",    # plus (filled)
    "bpgs":                 "o",    # circle
    "bpgs_canonical":       "o",
    "bpgs_batch_aware_fixed": "X",
    "bpgs_stateless_auto":  "p",    # pentagon
    "bpgs_stateless_fixed": "h",    # hexagon
}

LINESTYLES = {
    "static":               "-",
    "kendall":              "-",
    "uwso":                 "-",
    "pcgrad":               "-",
    "gradnorm_proxy":       "-",
    "bpgs":                 "-",
    "bpgs_canonical":       "-",
    "bpgs_batch_aware_fixed": "--",
    "bpgs_stateless_auto":  "-.",
    "bpgs_stateless_fixed": ":",
}

# Hatching for bar charts (optional)
HATCHES = {
    "static":               "",
    "kendall":              "",
    "uwso":                 "",
    "pcgrad":               "",
    "gradnorm_proxy":       "",
    "bpgs":                 "",
    "bpgs_canonical":       "",
    "bpgs_batch_aware_fixed": "//",
    "bpgs_stateless_auto":  "\\\\",
    "bpgs_stateless_fixed": "..",
}


NEURIPS_RC: dict = {
    # Fonts
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":            _FONT_SIZE_LABEL,
    "mathtext.fontset":     "cm",

    # Axes
    "axes.titlesize":       _FONT_SIZE_TITLE,
    "axes.labelsize":       _FONT_SIZE_LABEL,
    "axes.titleweight":     "normal",
    "axes.titlepad":        8,
    "axes.labelpad":        5,
    "axes.linewidth":       0.6,
    "axes.edgecolor":       "#333333",
    "axes.facecolor":       "white",
    "axes.grid":            True,
    "axes.grid.which":      "major",
    "axes.axisbelow":       True,
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # Grid
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.7,
    "grid.linestyle":       "-",

    # Ticks
    "xtick.labelsize":      _FONT_SIZE_TICK,
    "ytick.labelsize":      _FONT_SIZE_TICK,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.pad":      3,
    "ytick.major.pad":      3,

    # Legend
    "legend.fontsize":      _FONT_SIZE_LEGEND,
    "legend.frameon":        True,
    "legend.framealpha":     0.92,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fancybox":      True,
    "legend.borderpad":     0.5,
    "legend.handlelength":  1.8,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.labelspacing":  0.35,

    # Lines
    "lines.linewidth":      1.5,
    "lines.markersize":     5,

    # Figure
    "figure.facecolor":     "white",
    "figure.dpi":           _DPI,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,

    # Saving
    "savefig.dpi":          _DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.08,
    "savefig.facecolor":    "white",
    "savefig.transparent":  False,

    # PDF
    "pdf.fonttype":         42,
    "ps.fonttype":          42,

    # Error bars
    "errorbar.capsize":     2.5,
}


@contextlib.contextmanager
def apply_neurips_style():
    """Context manager that applies the NeurIPS publication style.

    All figures created within this block inherit the style. The
    previous style is restored on exit.

    Example::

        with apply_neurips_style():
            fig, ax = create_figure(width="single")
            ax.plot(x, y)
    """
    with mpl.rc_context(NEURIPS_RC):
        yield


def create_figure(
    width: str = "single",
    aspect: float = 0.618,
    nrows: int = 1,
    ncols: int = 1,
    height_override: Optional[float] = None,
    squeeze: bool = True,
) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, np.ndarray]]:
    """Create a figure with NeurIPS-appropriate dimensions.

    Parameters
    ----------
    width : {"single", "double"}
        Column width preset.
    aspect : float
        Height/width ratio (default: golden ratio).
    nrows, ncols : int
        Subplot grid dimensions.
    height_override : float, optional
        Explicit height in inches (overrides aspect).
    squeeze : bool
        Whether to squeeze singleton dimensions.

    Returns
    -------
    (fig, axes)
    """
    w = _TEXT_WIDTH if width == "double" else _COL_WIDTH
    h = height_override if height_override else w * aspect

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(w, h),
        squeeze=squeeze,
    )

    return fig, axes


def plot_with_seed_band(
    ax: plt.Axes,
    epochs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: Optional[str] = None,
    markevery: int = 10,
    linewidth: float = 1.4,
    alpha_band: float = 0.12,
) -> None:
    """Plot a mean line with a ±1σ shaded band.

    Parameters
    ----------
    ax : matplotlib Axes
    epochs : array of epoch numbers
    mean, std : arrays of metric values
    color : line color
    label : legend label
    linestyle : line dash pattern
    marker : marker shape (or None)
    markevery : plot marker every N points
    linewidth : line thickness
    alpha_band : opacity of the ±σ band
    """
    ax.plot(
        epochs, mean,
        color=color, label=label,
        linestyle=linestyle,
        marker=marker,
        markevery=markevery,
        markersize=4,
        linewidth=linewidth,
        zorder=3,
    )
    ax.fill_between(
        epochs,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha_band,
        linewidth=0,
        zorder=1,
    )


def add_legend_outside(
    ax: plt.Axes,
    loc: str = "upper right",
    ncol: int = 1,
    fontsize: Optional[float] = None,
) -> None:
    """Place legend outside the axes in a non-overlapping position.

    Parameters
    ----------
    ax : Axes
    loc : str
        Approximate location hint. The legend is placed *outside* the
        axes bounding box at the specified corner.
    ncol : int
        Number of columns.
    fontsize : float, optional
        Override default legend font size.
    """
    fs = fontsize or _FONT_SIZE_LEGEND

    loc_map = {
        "upper right": ("upper left", (1.02, 1.0)),
        "upper left":  ("upper right", (-0.02, 1.0)),
        "lower right": ("lower left", (1.02, 0.0)),
        "lower left":  ("lower right", (-0.02, 0.0)),
        "upper center": ("lower center", (0.5, 1.05)),
        "center right": ("center left", (1.02, 0.5)),
    }

    if loc in loc_map:
        mpl_loc, anchor = loc_map[loc]
        ax.legend(
            loc=mpl_loc,
            bbox_to_anchor=anchor,
            fontsize=fs,
            ncol=ncol,
            borderaxespad=0,
        )
    else:
        ax.legend(loc=loc, fontsize=fs, ncol=ncol)


def add_legend_top(
    fig: plt.Figure,
    ax: plt.Axes,
    ncol: int = 4,
    fontsize: Optional[float] = None,
) -> None:
    """Place a shared legend centered above the axes/figure.

    This avoids any overlap with plot content.
    """
    fs = fontsize or _FONT_SIZE_LEGEND
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        fontsize=fs,
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fancybox=True,
    )


def add_shared_legend_below(
    fig: plt.Figure,
    axes,
    ncol: int = 4,
    fontsize: Optional[float] = None,
) -> None:
    """Place a shared legend centered below the axes."""
    fs = fontsize or _FONT_SIZE_LEGEND
    # Collect handles from all axes
    handles, labels = [], []
    seen = set()
    if hasattr(axes, "flat"):
        ax_list = axes.flat
    elif isinstance(axes, (list, tuple)):
        ax_list = axes
    else:
        ax_list = [axes]

    for ax in ax_list:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)

    if not handles:
        return

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=fs,
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fancybox=True,
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
) -> None:
    """Save a figure to PDF and PNG in organized subdirectories.

    Creates ``pdf/`` and ``png/`` directories under ``path.parent``.

    Parameters
    ----------
    fig : matplotlib Figure
    path : Path
        Base path (without extension). e.g. ``output/my_figure``
    formats : sequence of str
        File formats to export.
    close : bool
        Close the figure after saving to free memory.
    """
    for fmt in formats:
        subdir = path.parent / fmt
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f"{path.name}.{fmt}"
        fig.savefig(str(out_path), format=fmt)
        logger.info("Saved figure: %s", out_path)

    if close:
        plt.close(fig)


def get_color(method: str) -> str:
    """Get the color for a method, with fallback."""
    return PALETTE.get(method, "#666666")


def get_marker(method: str) -> str:
    """Get the marker for a method, with fallback."""
    return MARKERS.get(method, "o")


def get_linestyle(method: str) -> str:
    """Get the linestyle for a method, with fallback."""
    return LINESTYLES.get(method, "-")


def method_plot_kwargs(method: str) -> dict:
    """Return a dict of color, marker, linestyle for a method.

    Convenient for ``ax.plot(..., **method_plot_kwargs("bpgs"))``.
    """
    return {
        "color": get_color(method),
        "marker": get_marker(method),
        "linestyle": get_linestyle(method),
    }
