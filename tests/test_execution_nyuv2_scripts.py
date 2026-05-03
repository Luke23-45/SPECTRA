from scripts.execution.experiment_runner import (
    EXPERIMENTS,
    METHODS_BY_DATASET,
    build_command,
    default_overrides_for_experiment,
)
from scripts.execution.reproduction_pipeline import (
    DEFAULT_NYUV2_SEEDS,
    NYUV2_PUBLICATION_METHODS,
    build_nyuv2_publication_plan,
    resolve_nyuv2_methods,
    resolve_seed_sweep,
)


def test_experiment_runner_registers_full_nyuv2_matrix():
    assert METHODS_BY_DATASET["nyuv2"] == [
        "static",
        "kendall",
        "uwso",
        "gradnorm_proxy",
        "pcgrad",
        "bpgs",
        "bpgs_alb",
    ]

    for method in METHODS_BY_DATASET["nyuv2"]:
        alias = f"{method}_nyuv2"
        assert alias in EXPERIMENTS
        assert default_overrides_for_experiment(alias) == [
            "dataset=nyuv2",
            f"method={method}",
        ]

    assert "nyuv2_generate" in EXPERIMENTS
    assert EXPERIMENTS["nyuv2_generate"]["type"] == "script"
    assert "rf1_generate" in EXPERIMENTS
    assert EXPERIMENTS["rf1_generate"]["type"] == "script"


def test_experiment_runner_registers_full_rf1_matrix():
    assert METHODS_BY_DATASET["rf1"] == [
        "static",
        "kendall",
        "uwso",
        "gradnorm_proxy",
        "pcgrad",
        "bpgs",
        "bpgs_alb",
    ]

    for method in METHODS_BY_DATASET["rf1"]:
        alias = f"{method}_rf1"
        assert alias in EXPERIMENTS
        assert default_overrides_for_experiment(alias) == [
            "dataset=rf1",
            f"method={method}",
        ]


def test_experiment_runner_registers_full_yeast_and_qm9_matrices():
    for dataset_name in ["yeast", "qm9"]:
        assert METHODS_BY_DATASET[dataset_name] == [
            "static",
            "kendall",
            "uwso",
            "gradnorm_proxy",
            "pcgrad",
            "bpgs",
            "bpgs_alb",
        ]
        for method in METHODS_BY_DATASET[dataset_name]:
            alias = f"{method}_{dataset_name}"
            assert alias in EXPERIMENTS
            assert default_overrides_for_experiment(alias) == [
                f"dataset={dataset_name}",
                f"method={method}",
            ]

    assert "yeast_generate" in EXPERIMENTS
    assert EXPERIMENTS["yeast_generate"]["type"] == "script"
    assert "qm9_generate" in EXPERIMENTS
    assert EXPERIMENTS["qm9_generate"]["type"] == "script"


def test_experiment_runner_builds_hydra_and_script_commands():
    hydra_cmd = build_command("kendall_nyuv2", extra_args=["seed=43"], epochs=200)
    assert hydra_cmd[1:3] == ["-m", "scripts.run_training"]
    assert "seed=43" in hydra_cmd
    assert "dataset=nyuv2" in hydra_cmd
    assert "method=kendall" in hydra_cmd
    assert "train.epochs=200" in hydra_cmd

    script_cmd = build_command("nyuv2_generate", extra_args=["--force"])
    assert script_cmd[1].endswith("scripts\\data\\nyuv2_generate.py") or script_cmd[1].endswith("scripts/data/nyuv2_generate.py")
    assert "--force" in script_cmd

    hydra_rf1_cmd = build_command("bpgs_rf1", extra_args=["seed=43"], epochs=200)
    assert "dataset=rf1" in hydra_rf1_cmd
    assert "method=bpgs" in hydra_rf1_cmd
    assert "train.epochs=200" in hydra_rf1_cmd

    rf1_script_cmd = build_command("rf1_generate", extra_args=["--force"])
    assert rf1_script_cmd[1].endswith("scripts\\data\\rf1_generate.py") or rf1_script_cmd[1].endswith("scripts/data/rf1_generate.py")
    assert "--force" in rf1_script_cmd

    qm9_cmd = build_command("bpgs_qm9", extra_args=["seed=43"], epochs=200)
    assert "dataset=qm9" in qm9_cmd
    assert "method=bpgs" in qm9_cmd

    yeast_cmd = build_command("uwso_yeast", extra_args=["seed=43"], epochs=20)
    assert "dataset=yeast" in yeast_cmd
    assert "method=uwso" in yeast_cmd

    yeast_script_cmd = build_command("yeast_generate", extra_args=["--force"])
    assert yeast_script_cmd[1].endswith("scripts\\data\\yeast_generate.py") or yeast_script_cmd[1].endswith("scripts/data/yeast_generate.py")
    qm9_script_cmd = build_command("qm9_generate", extra_args=["--force"])
    assert qm9_script_cmd[1].endswith("scripts\\data\\qm9_generate.py") or qm9_script_cmd[1].endswith("scripts/data/qm9_generate.py")


def test_resolve_nyuv2_methods_and_seeds_defaults():
    assert resolve_nyuv2_methods(None) == NYUV2_PUBLICATION_METHODS
    assert resolve_seed_sweep(None) == DEFAULT_NYUV2_SEEDS


def test_nyuv2_default_root_matches_lmdb_layout():
    import yaml

    with open("configs/dataset/nyuv2.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert cfg["root"] == "datasets/nyuv2_lmdb/data"
    assert cfg["dataset"]["root"] == "${root}"


def test_pcgrad_method_enables_resume_friendly_logging_and_checkpoints():
    import yaml

    with open("configs/method/pcgrad.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert cfg["train"]["save_ckpt"] is True
    assert cfg["train"]["checkpoint_every_minutes"] == 10
    assert cfg["logging"]["use_wandb"] is True
    assert cfg["logging"]["wandb_mode"] == "offline"


def test_build_nyuv2_publication_plan_includes_data_prep_and_seed_sweep():
    plan = build_nyuv2_publication_plan(
        methods=["static", "bpgs"],
        seeds=[42, 99],
        include_data_prep=True,
        data_prep_args=["--force"],
    )

    assert plan[0]["experiment"] == "nyuv2_generate"
    assert plan[0]["args"] == ["--force"]
    assert plan[1]["experiment"] == "static_nyuv2"
    assert plan[1]["args"] == ["seed=42"]
    assert plan[2]["experiment"] == "bpgs_nyuv2"
    assert plan[2]["args"] == ["seed=42"]
    assert plan[3]["experiment"] == "static_nyuv2"
    assert plan[3]["args"] == ["seed=99"]
    assert plan[4]["experiment"] == "bpgs_nyuv2"
    assert plan[4]["args"] == ["seed=99"]


def test_data_prep_stage_args_do_not_include_training_overrides():
    plan = build_nyuv2_publication_plan(
        methods=["static"],
        seeds=[42],
        include_data_prep=True,
        data_prep_args=["--force"],
    )

    data_stage_args = list(plan[0]["args"])
    training_extra = ["train.batch_size=4", "logging.use_wandb=true"]

    final_data_stage_args = list(data_stage_args)
    final_train_stage_args = list(plan[1]["args"]) + training_extra

    assert final_data_stage_args == ["--force"]
    assert final_train_stage_args == ["seed=42", "train.batch_size=4", "logging.use_wandb=true"]
