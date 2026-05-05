"""Main synthetic benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectra.baselines.kendall import KendallWeighter
from spectra.baselines.pcgrad import PCGradWeighter
from spectra.baselines.static import StaticWeighter
from spectra.baselines.uwso import UWSOWeighter
from spectra.core.bpgs import BPGS
from spectra.empirical.generators import (
    ConflictControlledGenerator,
    CorrelationControlledGenerator,
    HeterogeneousControlledGenerator,
    ImbalanceControlledGenerator,
    RescalingControlledGenerator,
    SyntheticGeneratorConfig,
    SyntheticProblem,
    TaskSpec,
    TensorTaskDataset,
)
from spectra.empirical.metrics import (
    compute_boundedness_metrics,
    compute_stability_metrics,
    compute_task_performance,
    summarize_performance,
)
from spectra.empirical.runners.base_runner import BaseRunner, ExperimentResult, RunnerConfig
from spectra.empirical.runners.seed_manager import apply_seed_bundle, build_seed_bundle
from spectra.utils.wandb import WandbSession, WandbSettings


GENERATOR_REGISTRY = {
    "correlation": CorrelationControlledGenerator,
    "imbalance": ImbalanceControlledGenerator,
    "conflict": ConflictControlledGenerator,
    "rescaling": RescalingControlledGenerator,
    "heterogeneous": HeterogeneousControlledGenerator,
}


class SharedTrunkNet(nn.Module):
    """Small but expressive shared-trunk multi-task model for controlled experiments."""

    def __init__(self, input_dim: int, hidden_dim: int, task_specs: Sequence[TaskSpec]) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict()
        for spec in task_specs:
            self.heads[spec.name] = nn.Linear(hidden_dim, spec.output_dim)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk(inputs)
        return {name: head(features) for name, head in self.heads.items()}


@dataclass
class SyntheticRunSpec:
    generator_config: SyntheticGeneratorConfig
    runner_config: RunnerConfig


class SyntheticExperimentRunner(BaseRunner):
    """Execute one synthetic empirical run for one method and one seed."""

    def __init__(self, spec: SyntheticRunSpec) -> None:
        super().__init__(spec.runner_config)
        self.spec = spec
        self._wandb_session = None

    def run(self) -> ExperimentResult:
        bundle = build_seed_bundle(self.config.seed)
        apply_seed_bundle(bundle)
        problem = self._build_problem()
        run_id = f"{self.config.family}-{self.config.method}-seed-{self.config.seed}"
        wandb_session = WandbSession(
            WandbSettings.from_logging_config(self.config.logging),
            self.result_dir,
            run_id,
            name=run_id,
            group=self.config.family,
            job_type="empirical-train",
            tags=(self.config.family, self.config.method, f"seed:{self.config.seed}"),
            config_payload={
                "runner": self.config.to_dict(),
                "generator": self.spec.generator_config.to_dict(),
                "problem": problem.to_metadata(),
            },
            summary_payload={
                "result_dir": str(self.result_dir),
                "family": self.config.family,
                "method": self.config.method,
                "seed": self.config.seed,
            },
        )
        self._wandb_session = wandb_session
        model = SharedTrunkNet(
            input_dim=problem.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            task_specs=problem.task_specs,
        ).to(self.config.device)
        train_loader = DataLoader(
            problem.train_dataset,
            batch_size=min(self.config.batch_size, len(problem.train_dataset)),
            shuffle=True,
            collate_fn=TensorTaskDataset.collate_fn,
        )
        val_loader = DataLoader(
            problem.val_dataset,
            batch_size=min(self.config.batch_size, len(problem.val_dataset)),
            shuffle=False,
            collate_fn=TensorTaskDataset.collate_fn,
        )
        try:
            history, predictions, targets = self._train(model, problem.task_specs, train_loader, val_loader)
            task_metrics = compute_task_performance(predictions, targets, problem.task_specs)
            final_metrics = summarize_performance(task_metrics)
            final_metrics.update(compute_stability_metrics(history))
            final_metrics.update(compute_boundedness_metrics(history))
            result = ExperimentResult(
                config={"runner": self.config.to_dict(), "generator": self.spec.generator_config.to_dict()},
                problem_metadata=problem.to_metadata(),
                final_metrics=final_metrics,
                task_metrics=task_metrics,
                history=history,
                method=self.config.method,
                seed=self.config.seed,
            )
            saved = self.save_result(result)
            wandb_session.log_metrics({f"final/{key}": value for key, value in final_metrics.items()})
            wandb_session.update_summary(
                {
                    "final_metrics": final_metrics,
                    "task_metrics": task_metrics,
                    "result_json": saved.artifacts.get("result_json"),
                    "history_csv": saved.artifacts.get("history_csv"),
                }
            )
            wandb_session.log_artifact(
                name=f"{run_id}-artifacts",
                files={
                    "result.json": Path(saved.artifacts["result_json"]),
                    "history.csv": Path(saved.artifacts["history_csv"]),
                },
            )
            return saved
        finally:
            wandb_session.finish()

    def _build_problem(self) -> SyntheticProblem:
        generator_cls = GENERATOR_REGISTRY[self.config.family]
        generator = generator_cls(self.spec.generator_config)
        return generator.generate()

    def _build_weighter(self, num_tasks: int) -> nn.Module:
        method = self.config.method.lower()
        if method == "static":
            return StaticWeighter(num_tasks=num_tasks)
        if method == "uwso":
            return UWSOWeighter(num_tasks=num_tasks)
        if method == "pcgrad":
            return PCGradWeighter(num_tasks=num_tasks)
        if method == "kendall":
            return KendallWeighter(num_tasks=num_tasks)
        if method == "bpgs":
            return BPGS(
                num_tasks=num_tasks,
                **self.config.bpgs_architecture,
            )
        raise ValueError(f"Unsupported method: {self.config.method}")

    def _criterion(self, spec: TaskSpec, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if spec.task_type == "regression":
            pred = prediction
            tgt = target.float()
            if spec.output_dim == 1 and pred.ndim > 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            if spec.output_dim == 1 and tgt.ndim > 1 and tgt.shape[-1] == 1:
                tgt = tgt.squeeze(-1)
            if spec.loss == "mae":
                return nn.functional.l1_loss(pred, tgt)
            if spec.loss == "smooth_l1":
                return nn.functional.smooth_l1_loss(pred, tgt)
            return nn.functional.mse_loss(pred, tgt)
        if spec.task_type == "binary":
            return nn.functional.binary_cross_entropy_with_logits(prediction.squeeze(-1), target.float())
        if spec.task_type == "multiclass":
            return nn.functional.cross_entropy(prediction, target.long())
        pred_unit = nn.functional.normalize(prediction, dim=-1)
        target_unit = nn.functional.normalize(target.float(), dim=-1)
        cosine = (pred_unit * target_unit).sum(dim=-1).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        if spec.loss == "cosine":
            return (1.0 - cosine).mean()
        return torch.acos(cosine).mean()

    def _apply_training_stress(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.config.loss_scale_task or float(self.config.loss_scale_multiplier) == 1.0:
            return losses
        scaled = dict(losses)
        if self.config.loss_scale_task in scaled:
            scaled[self.config.loss_scale_task] = scaled[self.config.loss_scale_task] * float(self.config.loss_scale_multiplier)
        return scaled

    def _compute_losses(
        self,
        model: nn.Module,
        batch: Mapping[str, object],
        task_specs: Sequence[TaskSpec],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        inputs = batch["input"].to(self.config.device)
        targets = {name: tensor.to(self.config.device) for name, tensor in batch["targets"].items()}
        outputs = model(inputs)
        losses = {spec.name: self._criterion(spec, outputs[spec.name], targets[spec.name]) for spec in task_specs}
        return losses, outputs

    def _gradient_summary(self, losses: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
        params = [param for param in model.parameters() if param.requires_grad]
        task_names = list(losses.keys())
        flat_grads: List[torch.Tensor] = []
        for name in task_names:
            grads = torch.autograd.grad(losses[name], params, retain_graph=True, allow_unused=True)
            flat = []
            for grad, param in zip(grads, params):
                if grad is None:
                    flat.append(torch.zeros_like(param).reshape(-1))
                else:
                    flat.append(grad.reshape(-1))
            flat_grads.append(torch.cat(flat))

        norms = [grad.norm().item() for grad in flat_grads]
        cosines: List[float] = []
        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                denom = (flat_grads[i].norm() * flat_grads[j].norm()).item()
                cosines.append(0.0 if denom < 1e-12 else torch.dot(flat_grads[i], flat_grads[j]).item() / denom)
        return {
            "mean_grad_norm": float(np.mean(norms)),
            "max_grad_norm": float(np.max(norms)),
            "mean_pairwise_cosine": float(np.mean(cosines) if cosines else 1.0),
            "min_pairwise_cosine": float(np.min(cosines) if cosines else 1.0),
        }

    def _train(
        self,
        model: nn.Module,
        task_specs: Sequence[TaskSpec],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        method = self.config.method.lower()
        weighter = self._build_weighter(len(task_specs)).to(self.config.device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        aux_optimizer = None
        if method == "kendall":
            model_optimizer = torch.optim.Adam(
                list(model.parameters()) + list(weighter.parameters()),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif method == "bpgs":
            aux_optimizer = torch.optim.Adam(weighter.parameters(), lr=self.config.learning_rate)

        history: List[Dict[str, object]] = []
        for epoch in range(self.config.epochs):
            model.train()
            epoch_losses: List[float] = []
            gradient_summary: Dict[str, float] | None = None
            last_task_weights: Dict[str, float] = {}
            last_latent_state: Dict[str, float] = {}

            for batch_index, batch in enumerate(train_loader):
                losses, _ = self._compute_losses(model, batch, task_specs)
                scaled_losses = self._apply_training_stress(losses)
                if gradient_summary is None:
                    gradient_summary = self._gradient_summary(scaled_losses, model)
                task_names = [spec.name for spec in task_specs]
                losses_tensor = torch.stack([scaled_losses[name] for name in task_names])

                if method == "pcgrad":
                    params = [param for param in model.parameters() if param.requires_grad]
                    task_grads = []
                    for index, name in enumerate(task_names):
                        grads = torch.autograd.grad(
                            scaled_losses[name],
                            params,
                            retain_graph=index < len(task_names) - 1,
                            allow_unused=True,
                        )
                        task_grads.append(
                            [
                                torch.zeros_like(param) if grad is None else grad.detach().clone()
                                for grad, param in zip(grads, params)
                            ]
                        )
                    model_optimizer.zero_grad(set_to_none=True)
                    metrics = weighter.project_and_assign(task_grads, params)
                    nn.utils.clip_grad_norm_(params, self.config.gradient_clip_norm)
                    model_optimizer.step()
                    total_loss = losses_tensor.sum().detach()
                    last_task_weights = {f"task_{i}": 1.0 / len(task_specs) for i in range(len(task_specs))}
                    last_latent_state = {"pcgrad/total_conflicts": float(metrics["pcgrad/total_conflicts"])}
                elif method == "bpgs":
                    model_optimizer.zero_grad(set_to_none=True)
                    loss_list = [scaled_losses[name] for name in task_names]
                    loss_net = weighter.network_loss(loss_list)
                    loss_net.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    model_optimizer.step()

                    if aux_optimizer is None:
                        raise RuntimeError("BPGS requires an auxiliary optimizer.")
                    aux_optimizer.zero_grad(set_to_none=True)
                    loss_unc = weighter.uncertainty_loss(loss_list)
                    loss_unc.backward()
                    nn.utils.clip_grad_norm_(weighter.parameters(), self.config.gradient_clip_norm)
                    aux_optimizer.step()
                    total_loss = loss_net.detach()
                    with torch.no_grad():
                        detached_s = weighter.get_s(loss_list).detach().cpu()
                        detached_weights = torch.exp(-detached_s)
                        detached_theta = weighter.theta.detach().cpu()
                    last_task_weights = {
                        spec.name: float(value.item())
                        for spec, value in zip(task_specs, detached_weights)
                    }
                    last_latent_state = {f"s_{index}": float(value.item()) for index, value in enumerate(detached_s)}
                    last_latent_state.update({
                        f"theta_{index}": float(value.item())
                        for index, value in enumerate(detached_theta)
                    })
                else:
                    model_optimizer.zero_grad(set_to_none=True)
                    total_loss, metrics = weighter(losses_tensor)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    model_optimizer.step()
                    epoch_metrics = {key: value for key, value in metrics.items() if "weight_" in key}
                    last_task_weights = {
                        spec.name: float(epoch_metrics.get(f"{method}/weight_{index}", 1.0 / len(task_specs)))
                        for index, spec in enumerate(task_specs)
                    }
                    if method == "kendall":
                        last_latent_state = {f"log_var_{index}": float(weighter.log_vars[index].detach().cpu().item()) for index in range(len(task_specs))}
                    else:
                        last_latent_state = {}

                epoch_losses.append(float(total_loss.cpu().item()))

            val_summary = self._evaluate(model, task_specs, val_loader)
            epoch_record = {
                "epoch": epoch,
                "train_total_loss": float(np.mean(epoch_losses)),
                "val_total_loss": float(val_summary["mean_loss"]),
                "gradient_summary": gradient_summary or {},
                "task_weights": last_task_weights,
                "latent_state": last_latent_state,
                "architecture_diagnostics": {},
                "task_loss_summary": val_summary["task_loss_summary"],
            }
            history.append(epoch_record)
            if self._wandb_session is not None:
                self._wandb_session.log_metrics(
                    {
                        "epoch": epoch,
                        "train_total_loss": epoch_record["train_total_loss"],
                        "val_total_loss": epoch_record["val_total_loss"],
                        "gradient_summary": epoch_record["gradient_summary"],
                        "task_weights": epoch_record["task_weights"],
                        "latent_state": epoch_record["latent_state"],
                        "task_loss_summary": epoch_record["task_loss_summary"],
                    },
                    step=epoch,
                )

        predictions, targets = self._collect_predictions(model, task_specs, val_loader)
        return history, predictions, targets

    def _evaluate(
        self,
        model: nn.Module,
        task_specs: Sequence[TaskSpec],
        data_loader: DataLoader,
    ) -> Dict[str, object]:
        model.eval()
        batch_losses: List[float] = []
        task_loss_accumulator = {spec.name: [] for spec in task_specs}
        with torch.no_grad():
            for batch in data_loader:
                losses, _ = self._compute_losses(model, batch, task_specs)
                for name, loss in losses.items():
                    task_loss_accumulator[name].append(float(loss.cpu().item()))
                batch_losses.append(float(torch.stack(list(losses.values())).mean().cpu().item()))
        return {
            "mean_loss": float(np.mean(batch_losses)),
            "task_loss_summary": {name: float(np.mean(values)) for name, values in task_loss_accumulator.items()},
        }

    def _collect_predictions(
        self,
        model: nn.Module,
        task_specs: Sequence[TaskSpec],
        data_loader: DataLoader,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        model.eval()
        preds = {spec.name: [] for spec in task_specs}
        targets = {spec.name: [] for spec in task_specs}
        with torch.no_grad():
            for batch in data_loader:
                losses, outputs = self._compute_losses(model, batch, task_specs)
                batch_targets = batch["targets"]
                for spec in task_specs:
                    tensor = outputs[spec.name].detach().cpu()
                    if spec.task_type == "multiclass":
                        preds[spec.name].append(tensor.numpy())
                    elif tensor.ndim > 1 and tensor.shape[-1] == 1:
                        preds[spec.name].append(tensor.squeeze(-1).numpy())
                    else:
                        preds[spec.name].append(tensor.numpy())
                    targets[spec.name].append(batch_targets[spec.name].numpy())
        return (
            {name: np.concatenate(values, axis=0) for name, values in preds.items()},
            {name: np.concatenate(values, axis=0) for name, values in targets.items()},
        )
