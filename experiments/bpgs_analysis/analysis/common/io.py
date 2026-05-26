"""
Data I/O utilities for the BPGS analysis framework.

Handles all file loading and path discovery across the heterogeneous
directory layouts of the five experiment studies (ablation, NYUv2,
full-data, stress-scale, stress-heterogeneous).

Key responsibilities
--------------------
1. **CSV parsing** – Correctly merge the sparse multi-row-per-epoch
   layout produced by the PyTorch Lightning CSV logger.
2. **Path discovery** – Resolve run directories despite inconsistent
   nesting patterns (double-nested NYUv2, run-id-based ablation, etc.).
3. **JSON/YAML loading** – Parse ``run_summary.json``, ``metadata.json``,
   ``config.yaml``, and stress ``results.csv``/``summary.json``.

Design notes
------------
* All public functions accept ``pathlib.Path`` objects and perform
  existence checks with informative error messages.
* ``discover_runs`` uses pluggable *path resolver* callables so each
  study can define its own naming convention without changing the core
  logic.
* The CSV parser uses ``pandas.groupby().last()`` which, by pandas
  convention, returns the last **non-null** value per column within
  each group — exactly the semantics we need for the sparse layout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import yaml

from analysis.common.constants import (
    ABLATION_BASE_DIR,
    ABLATION_METHODS,
    ABLATION_RUN_PREFIX,
    FULL_DATA_METHODS,
    NYUV2_BASE_DIR,
    NYUV2_DOUBLE_NESTED,
    NYUV2_METHODS,
    RF1_BASE_DIR,
    SEEDS,
    STRESS_HETERO_BASE_DIR,
    STRESS_HETERO_METHODS,
    STRESS_IGNORE_DIRS,
    STRESS_REGIMES,
    STRESS_RESCALE_BASE_DIR,
    STRESS_SCALE_BASE_DIR,
    STRESS_SCALE_METHODS,
    STRESS_SCALES,
    YEAST_BASE_DIR,
)

logger = logging.getLogger(__name__)

# Type alias for path resolver functions
PathResolver = Callable[[Path, str, int], Path]


@dataclass
class RunInfo:
    """Metadata for a single experiment run directory.

    Attributes
    ----------
    method : str
        Method identifier (e.g. ``"bpgs"``, ``"kendall"``).
    seed : int
        Random seed value.
    path : Path
        Absolute or resolved path to the run directory.
    exists : bool
        Whether the path exists on disk.
    """

    method: str
    seed: int
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists() and self.path.is_dir()

    def __repr__(self) -> str:
        status = "✓" if self.exists else "✗"
        return f"RunInfo({status} {self.method}/s{self.seed} @ {self.path})"


@dataclass
class DiscoveryReport:
    """Summary of a ``discover_runs`` call.

    Attributes
    ----------
    runs : dict
        ``method → seed → RunInfo`` mapping.
    found : int
        Number of runs whose paths exist on disk.
    total : int
        Total number of expected runs.
    missing : list of str
        Human-readable list of missing runs.
    """

    runs: Dict[str, Dict[int, RunInfo]]
    found: int = 0
    total: int = 0
    missing: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True if every expected run was found."""
        return self.found == self.total

    def raise_if_incomplete(self, study_name: str = "study") -> None:
        """Raise ``FileNotFoundError`` if any runs are missing."""
        if not self.complete:
            msg = (
                f"Incomplete {study_name} data: found {self.found}/{self.total} runs. "
                f"Missing: {self.missing}"
            )
            raise FileNotFoundError(msg)

    def warn_if_incomplete(self, study_name: str = "study") -> None:
        """Log a warning if any runs are missing."""
        if not self.complete:
            logger.warning(
                "Incomplete %s data: found %d/%d runs. Missing: %s",
                study_name,
                self.found,
                self.total,
                self.missing,
            )


def load_json(path: Path) -> Any:
    """Load a JSON file with informative error on failure.

    Parameters
    ----------
    path : Path
        Path to a ``.json`` file.

    Returns
    -------
    Any
        Parsed JSON content (dict, list, etc.).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file.

    Parameters
    ----------
    path : Path
        Path to a ``.yaml`` or ``.yml`` file.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_metrics_csv(run_dir: Path) -> Path:
    """Locate the ``metrics.csv`` inside a training run directory.

    The CSV lives at ``run_dir/csv_logs/<run_id>/metrics.csv`` where
    ``<run_id>`` varies per run.  We glob for the first match.

    Parameters
    ----------
    run_dir : Path
        Root directory of a training run (contains ``csv_logs/``).

    Returns
    -------
    Path
        Absolute path to the ``metrics.csv`` file.

    Raises
    ------
    FileNotFoundError
        If ``csv_logs/`` or ``metrics.csv`` cannot be found.
    """
    csv_logs = run_dir / "csv_logs"
    if not csv_logs.exists():
        raise FileNotFoundError(
            f"csv_logs/ directory not found in run dir: {run_dir}"
        )

    # Expect exactly one run_id subdirectory
    subdirs = sorted(
        [d for d in csv_logs.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No run_id subdirectory found in {csv_logs}. "
            f"Contents: {list(csv_logs.iterdir())}"
        )
    if len(subdirs) > 1:
        logger.warning(
            "Multiple run_id directories in %s; using '%s'. All: %s",
            csv_logs,
            subdirs[0].name,
            [d.name for d in subdirs],
        )

    metrics_csv = subdirs[0] / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"metrics.csv not found at expected path: {metrics_csv}"
        )
    return metrics_csv


def load_metrics_csv(run_dir: Path) -> pd.DataFrame:
    """Load and parse a sparse multi-row-per-epoch metrics CSV.

    The PyTorch Lightning CSV logger writes multiple rows per epoch:
    some contain only step-level metrics (learning rate, step loss),
    some contain only epoch-level validation metrics, and some contain
    only BPGS internal state (theta, weights, log_var).

    **Strategy**:

    1. Read the CSV with ``pandas.read_csv()`` (handles ``\\r\\n``).
    2. Forward-fill the ``epoch`` column so that intermediate step rows
       inherit their parent epoch.
    3. Group by ``epoch`` and take the **last non-null** value per
       column using ``groupby().last()`` (pandas convention).
    4. Return one clean row per epoch with all metrics merged.

    Parameters
    ----------
    run_dir : Path
        Root of a training run (e.g.
        ``.../bpgs_canonical/bpgs_nyuv2_s42/``).

    Returns
    -------
    pd.DataFrame
        One row per epoch, indexed by integer epoch number.
        Columns are the union of all logged metrics.
    """
    csv_path = _find_metrics_csv(run_dir)

    # Read CSV — pandas handles \r\n (Windows) line endings natively
    df = pd.read_csv(csv_path)

    if df.empty:
        logger.warning("Empty metrics CSV: %s", csv_path)
        return df

    if "epoch" not in df.columns:
        raise ValueError(
            f"metrics.csv is missing required 'epoch' column: {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Forward-fill epoch for intermediate step rows
    df["epoch"] = df["epoch"].ffill()

    # Drop rows without an epoch (possible at the very start of training)
    df = df.dropna(subset=["epoch"])
    df["epoch"] = df["epoch"].astype(int)

    # Group by epoch → last non-null value per column.
    epoch_df = df.groupby("epoch", sort=True).last()

    epoch_df = epoch_df.reset_index()

    logger.debug(
        "Loaded metrics CSV: %s → %d epochs, %d columns",
        csv_path.parent.name,
        len(epoch_df),
        len(epoch_df.columns),
    )
    return epoch_df


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Load ``run_summary.json`` from a training run directory.

    Parameters
    ----------
    run_dir : Path
        Root directory of the training run.

    Returns
    -------
    dict
        Keys include ``fit_started_at``, ``elapsed_seconds``,
        ``stopped_epoch``, ``selection_metric``, ``selection_mode``,
        ``selected_checkpoint``, ``checkpoint_registry``.
    """
    return load_json(run_dir / "run_summary.json")


def load_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load ``metadata.json`` from a training run directory.

    Parameters
    ----------
    run_dir : Path
        Root directory of the training run.

    Returns
    -------
    dict
        Keys include ``experiment``, ``system``, ``dataset``,
        ``training``, ``tasks``, ``paths``, ``timestamp``.
    """
    return load_json(run_dir / "metadata.json")


def load_stress_results(seed_dir: Path) -> pd.DataFrame:
    """Load ``metrics/results.csv`` from a stress-test seed directory.

    Unlike training CSVs, stress ``results.csv`` files are dense
    (one row per seed) and do not require sparse-row merging.

    Parameters
    ----------
    seed_dir : Path
        Seed-level directory (e.g.
        ``.../x10/bpgs/seed_42/``).

    Returns
    -------
    pd.DataFrame
        Typically one row with columns: ``method``, ``seed``,
        ``macro_score``, ``worst_task_score``, etc.
    """
    csv_path = seed_dir / "metrics" / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stress results.csv not found: {csv_path}"
        )
    df = pd.read_csv(csv_path)

    # Drop empty trailing rows (some CSVs have a trailing newline)
    df = df.dropna(how="all")

    logger.debug(
        "Loaded stress results: %s → %d rows",
        csv_path,
        len(df),
    )
    return df


def load_stress_summary(seed_dir: Path) -> Dict[str, Any]:
    """Load ``metrics/summary.json`` from a stress-test seed directory.

    Parameters
    ----------
    seed_dir : Path
        Seed-level directory.

    Returns
    -------
    dict or list
        Aggregated summary (mean, std, count, stderr).
    """
    return load_json(seed_dir / "metrics" / "summary.json")


def resolve_ablation_path(base_dir: Path, method: str, seed: int) -> Path:
    """Resolve an ablation run directory.

    BPGS variants use ``{variant}/bpgs_nyuv2_s{seed}/`` (note: prefix
    is always ``bpgs``, not the variant name).  Kendall uses
    ``kendall/kendall_nyuv2_s{seed}/``.
    """
    prefix = ABLATION_RUN_PREFIX.get(method, method)
    return base_dir / method / f"{prefix}_nyuv2_s{seed}"


def resolve_nyuv2_path(base_dir: Path, method: str, seed: int) -> Path:
    """Resolve a NYUv2 benchmark run directory.

    Two nesting patterns exist:

    - **Double-nested**: ``bpgs/bpgs/seed_42/``, ``static/static/seed_42/``
    - **Single-nested**: ``kendall/seed_42/``, ``uwso/seed_42/``

    We try double-nested first, then fall back to single-nested.
    """
    if method in NYUV2_DOUBLE_NESTED:
        candidate = base_dir / method / method / f"seed_{seed}"
        if candidate.exists():
            return candidate
    return base_dir / method / f"seed_{seed}"


def resolve_full_data_path(base_dir: Path, method: str, seed: int) -> Path:
    """Resolve a full-data (Yeast / RF1) run directory.

    Consistent pattern: ``{method}/seed_{seed}/``.
    """
    return base_dir / method / f"seed_{seed}"


def discover_runs(
    base_dir: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    path_resolver: PathResolver,
) -> DiscoveryReport:
    """Discover experiment run directories using a pluggable resolver.

    Parameters
    ----------
    base_dir : Path
        Root directory for the study (e.g.
        ``data_root / "abalation/final_outputs"``).
    methods : sequence of str
        Method names to search for.
    seeds : sequence of int
        Seed values to search for.
    path_resolver : callable
        ``(base_dir, method, seed) → Path`` that returns the expected
        run directory path for a given (method, seed) pair.

    Returns
    -------
    DiscoveryReport
        Contains the ``runs`` mapping and completeness metadata.
    """
    runs: Dict[str, Dict[int, RunInfo]] = {}
    missing: List[str] = []
    found = 0
    total = 0

    for method in methods:
        runs[method] = {}
        for seed in seeds:
            run_dir = path_resolver(base_dir, method, seed)
            info = RunInfo(method=method, seed=seed, path=run_dir)
            runs[method][seed] = info
            total += 1

            if info.exists:
                found += 1
            else:
                missing.append(f"{method}/seed_{seed}")

    report = DiscoveryReport(
        runs=runs,
        found=found,
        total=total,
        missing=missing,
    )

    logger.info(
        "Discovered %d/%d runs in %s%s",
        found,
        total,
        base_dir,
        f" (missing: {missing})" if missing else "",
    )
    return report


def discover_ablation_runs(
    data_root: Path,
    methods: Sequence[str] = ABLATION_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> DiscoveryReport:
    """Discover all ablation study runs.

    Parameters
    ----------
    data_root : Path
        Root of ``bpgs_experiment/data/`` directory.
    methods : sequence of str
        Ablation method names.
    seeds : sequence of int
        Seed values.

    Returns
    -------
    DiscoveryReport
    """
    return discover_runs(
        data_root / ABLATION_BASE_DIR,
        methods,
        seeds,
        resolve_ablation_path,
    )


def discover_nyuv2_runs(
    data_root: Path,
    methods: Sequence[str] = NYUV2_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> DiscoveryReport:
    """Discover all NYUv2 benchmark runs."""
    return discover_runs(
        data_root / NYUV2_BASE_DIR,
        methods,
        seeds,
        resolve_nyuv2_path,
    )


def discover_full_data_runs(
    data_root: Path,
    experiment: str,
    methods: Sequence[str] = FULL_DATA_METHODS,
    seeds: Sequence[int] = SEEDS,
) -> DiscoveryReport:
    """Discover runs for a full-data experiment (Yeast or RF1).

    Parameters
    ----------
    data_root : Path
        Root of ``bpgs_experiment/data/``.
    experiment : str
        Either ``"yeast"`` or ``"rf1"``.
    """
    base_dirs = {
        "yeast": YEAST_BASE_DIR,
        "rf1":   RF1_BASE_DIR,
    }
    if experiment not in base_dirs:
        raise ValueError(
            f"Unknown full-data experiment '{experiment}'. "
            f"Expected one of: {list(base_dirs.keys())}"
        )
    return discover_runs(
        data_root / base_dirs[experiment],
        methods,
        seeds,
        resolve_full_data_path,
    )


def discover_stress_scale_runs(
    data_root: Path,
    experiment: str = "scale",
    methods: Sequence[str] = STRESS_SCALE_METHODS,
    seeds: Sequence[int] = SEEDS,
    scales: Sequence[int] = STRESS_SCALES,
) -> Dict[int, DiscoveryReport]:
    """Discover runs for a stress scale experiment.

    Parameters
    ----------
    data_root : Path
        Root of ``bpgs_experiment/data/``.
    experiment : str
        ``"scale"`` for 02_synthetic_scale_stress,
        ``"rescaling"`` for 06_pure_loss_rescaling.
    methods : sequence of str
        Stress method names.
    seeds : sequence of int
        Seed values.
    scales : sequence of int
        Scale factors (e.g. 1, 10, 100, 1000).

    Returns
    -------
    dict[int, DiscoveryReport]
        ``scale → DiscoveryReport`` mapping.
    """
    base_dirs = {
        "scale":     STRESS_SCALE_BASE_DIR,
        "rescaling": STRESS_RESCALE_BASE_DIR,
    }
    if experiment not in base_dirs:
        raise ValueError(
            f"Unknown stress scale experiment '{experiment}'. "
            f"Expected one of: {list(base_dirs.keys())}"
        )
    base = data_root / base_dirs[experiment]
    reports: Dict[int, DiscoveryReport] = {}

    for scale in scales:
        scale_dir = base / f"x{scale}"

        def _resolver(
            _base: Path,
            method: str,
            seed: int,
            _scale_dir: Path = scale_dir,
        ) -> Path:
            return _scale_dir / method / f"seed_{seed}"

        reports[scale] = discover_runs(
            scale_dir,
            methods,
            seeds,
            _resolver,
        )

    return reports


def discover_stress_hetero_runs(
    data_root: Path,
    methods: Sequence[str] = STRESS_HETERO_METHODS,
    seeds: Sequence[int] = SEEDS,
    regimes: Sequence[str] = STRESS_REGIMES,
) -> Dict[str, DiscoveryReport]:
    """Discover runs for the heterogeneous mixed-stress experiment.

    Parameters
    ----------
    data_root : Path
        Root of ``bpgs_experiment/data/``.
    methods : sequence of str
        Method names.
    seeds : sequence of int
        Seed values.
    regimes : sequence of str
        Regime names (``"clean"``, ``"noisy"``, ``"conflict"``).

    Returns
    -------
    dict[str, DiscoveryReport]
        ``regime → DiscoveryReport`` mapping.
    """
    base = data_root / STRESS_HETERO_BASE_DIR
    reports: Dict[str, DiscoveryReport] = {}

    for regime in regimes:
        regime_dir = base / regime

        def _resolver(
            _base: Path,
            method: str,
            seed: int,
            _regime_dir: Path = regime_dir,
        ) -> Path:
            return _regime_dir / method / f"seed_{seed}"

        reports[regime] = discover_runs(
            regime_dir,
            methods,
            seeds,
            _resolver,
        )

    return reports


def load_all_training_metrics(
    report: DiscoveryReport,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Load metrics CSVs for all runs in a discovery report.

    Parameters
    ----------
    report : DiscoveryReport
        Output of a ``discover_*`` function.

    Returns
    -------
    dict[str, dict[int, DataFrame]]
        ``method → seed → DataFrame`` with epoch-level metrics.
        Missing or failed runs are omitted with a warning.
    """
    all_metrics: Dict[str, Dict[int, pd.DataFrame]] = {}

    for method, seed_map in report.runs.items():
        all_metrics[method] = {}
        for seed, info in seed_map.items():
            if not info.exists:
                logger.warning("Skipping missing run: %s", info)
                continue
            try:
                df = load_metrics_csv(info.path)
                all_metrics[method][seed] = df
            except Exception as exc:
                logger.error(
                    "Failed to load metrics for %s seed=%d: %s",
                    method,
                    seed,
                    exc,
                )

    return all_metrics


def load_all_stress_results(
    report: DiscoveryReport,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Load stress ``results.csv`` for all runs in a discovery report.

    Parameters
    ----------
    report : DiscoveryReport
        Output of ``discover_stress_*``.

    Returns
    -------
    dict[str, dict[int, DataFrame]]
        ``method → seed → DataFrame`` (typically 1 row per seed).
    """
    all_results: Dict[str, Dict[int, pd.DataFrame]] = {}

    for method, seed_map in report.runs.items():
        all_results[method] = {}
        for seed, info in seed_map.items():
            if not info.exists:
                logger.warning("Skipping missing stress run: %s", info)
                continue
            try:
                df = load_stress_results(info.path)
                all_results[method][seed] = df
            except Exception as exc:
                logger.error(
                    "Failed to load stress results for %s seed=%d: %s",
                    method,
                    seed,
                    exc,
                )

    return all_results


def resolve_data_root(
    analysis_dir: Optional[Path] = None,
) -> Path:
    """Resolve the ``bpgs_experiment/data/`` root directory.

    First attempts to load the path from ``configs/base.yaml`` via
    the config module.  Falls back to auto-detection by walking up
    from this file if the config module is not available.

    Parameters
    ----------
    analysis_dir : Path, optional
        Path to the ``analysis/`` directory.  Defaults to auto-detect.

    Returns
    -------
    Path
        Absolute path to ``bpgs_experiment/data/``.

    Raises
    ------
    FileNotFoundError
        If the root cannot be found.
    """
    # Use the centralized config
    try:
        from analysis.common.config import get_data_root
        data_root = get_data_root()
        if data_root.exists():
            logger.debug("Data root from config: %s", data_root)
            return data_root
    except Exception:
        pass

    # Walk up from this file (legacy fallback)
    if analysis_dir is None:
        analysis_dir = Path(__file__).resolve().parent.parent

    data_root = analysis_dir.parent / "data"
    if not (data_root / "abalation").exists():
        raise FileNotFoundError(
            f"Cannot locate bpgs_experiment/data root. "
            f"Expected 'abalation/' in {data_root}. "
            f"Pass the correct path via analysis_dir parameter or "
            f"set paths.data_root in configs/base.yaml."
        )

    logger.debug("Data root resolved (fallback): %s", data_root)
    return data_root
