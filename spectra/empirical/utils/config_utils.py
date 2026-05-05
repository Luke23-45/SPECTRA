"""Hydra/OmegaConf adapters for empirical experiments."""

from __future__ import annotations

from typing import Dict

from omegaconf import DictConfig, OmegaConf

from spectra.empirical.generators import SyntheticGeneratorConfig
from spectra.empirical.runners.base_runner import RunnerConfig
from spectra.empirical.runners.method_comparison import ComparisonConfig


def validate_empirical_config(cfg: DictConfig) -> None:
    required_top_level = ("experiment", "family", "methods", "runtime", "output", "seeds")
    for key in required_top_level:
        if key not in cfg:
            raise ValueError(f"Empirical config missing required key: {key}")
    if "generator" not in cfg.experiment:
        raise ValueError("experiment.generator must be defined.")
    if not cfg.methods["names"]:
        raise ValueError("methods.names must be non-empty.")
    if not cfg.seeds["values"]:
        raise ValueError("seeds.values must be non-empty.")


def build_generator_config(cfg: DictConfig) -> SyntheticGeneratorConfig:
    payload = OmegaConf.to_container(cfg.experiment.generator, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("generator config must resolve to a mapping.")
    return SyntheticGeneratorConfig(**payload)


def build_comparison_config(cfg: DictConfig, output_dir: str) -> ComparisonConfig:
    bpgs_architecture = OmegaConf.to_container(cfg.get("bpgs", {}).get("architecture", {}), resolve=True)
    if not isinstance(bpgs_architecture, dict):
        bpgs_architecture = {}
    training_stress = cfg.experiment.get("training_stress", {})
    loss_scale_task = training_stress.get("loss_scale_task") if training_stress else None
    loss_scale_multiplier = float(training_stress.get("loss_scale_multiplier", 1.0)) if training_stress else 1.0
    return ComparisonConfig(
        family=str(cfg.family.name),
        methods=list(cfg.methods["names"]),
        seeds=[int(seed) for seed in cfg.seeds["values"]],
        epochs=int(cfg.runtime.epochs),
        batch_size=int(cfg.runtime.batch_size),
        hidden_dim=int(cfg.runtime.hidden_dim),
        learning_rate=float(cfg.runtime.learning_rate),
        weight_decay=float(cfg.runtime.weight_decay),
        device=str(cfg.runtime.device),
        output_dir=output_dir,
        bpgs_architecture=bpgs_architecture,
        loss_scale_task=str(loss_scale_task) if loss_scale_task not in (None, "") else None,
        loss_scale_multiplier=loss_scale_multiplier,
        logging=OmegaConf.to_container(cfg.get("logging", {}), resolve=True),
    )


def build_runner_config(cfg: DictConfig, method: str, seed: int, output_dir: str) -> RunnerConfig:
    bpgs_architecture = OmegaConf.to_container(cfg.get("bpgs", {}).get("architecture", {}), resolve=True)
    if not isinstance(bpgs_architecture, dict):
        bpgs_architecture = {}
    training_stress = cfg.experiment.get("training_stress", {})
    loss_scale_task = training_stress.get("loss_scale_task") if training_stress else None
    loss_scale_multiplier = float(training_stress.get("loss_scale_multiplier", 1.0)) if training_stress else 1.0
    return RunnerConfig(
        family=str(cfg.family.name),
        method=method,
        seed=seed,
        epochs=int(cfg.runtime.epochs),
        batch_size=int(cfg.runtime.batch_size),
        hidden_dim=int(cfg.runtime.hidden_dim),
        learning_rate=float(cfg.runtime.learning_rate),
        weight_decay=float(cfg.runtime.weight_decay),
        gradient_clip_norm=float(cfg.runtime.gradient_clip_norm),
        device=str(cfg.runtime.device),
        output_dir=output_dir,
        bpgs_architecture=bpgs_architecture,
        loss_scale_task=str(loss_scale_task) if loss_scale_task not in (None, "") else None,
        loss_scale_multiplier=loss_scale_multiplier,
        logging=OmegaConf.to_container(cfg.get("logging", {}), resolve=True),
    )
