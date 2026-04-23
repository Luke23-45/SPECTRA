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


def test_resolve_nyuv2_methods_and_seeds_defaults():
    assert resolve_nyuv2_methods(None) == NYUV2_PUBLICATION_METHODS
    assert resolve_seed_sweep(None) == DEFAULT_NYUV2_SEEDS


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
