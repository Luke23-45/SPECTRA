from omegaconf import OmegaConf

from spectra.utils.config import _merge_dataset_defaults


def test_dataset_defaults_do_not_override_top_level_overrides():
    cfg = OmegaConf.create(
        {
            "model": {"d_model": 128, "dropout": 0.1},
            "train": {"precision": "32", "lr": 1e-3},
            "dataset": {
                "name": "clinical",
                "model": {"d_model": 512, "dropout": 0.2},
                "train": {"precision": "16-mixed", "lr": 1e-4},
            },
        }
    )

    merged_model = OmegaConf.merge(cfg.model, cfg.dataset.model)
    merged_train = OmegaConf.merge(cfg.train, cfg.dataset.train)

    # Dataset values (used as second arg override) must win.
    assert merged_model.d_model == 512
    assert merged_model.dropout == 0.2
    assert merged_train.precision == "16-mixed"
    assert merged_train.lr == 1e-4


def test_struct_safe_merge_preserves_top_level_extra_keys():
    # Reproduces observed crash: dataset.model missing `n_heads` while
    # top-level model includes it via base config / CLI.
    dataset_model = OmegaConf.create({"backbone": "shared_trunk", "d_model": 512})
    top_model = OmegaConf.create({"backbone": "shared_trunk", "d_model": 128, "n_heads": 8})

    # Function signature: _merge_dataset_defaults(base_cfg, override_cfg)
    # where base_cfg = CLI/top-level, override_cfg = dataset defaults
    # Implementation: merge(defaults, cli_overrides) so CLI wins
    merged = _merge_dataset_defaults(top_model, dataset_model)
    assert merged.d_model == 128  # CLI (top_model) wins
    assert merged.n_heads == 8


def test_struct_safe_merge_is_deep_for_nested_sections():
    dataset_train = OmegaConf.create({"precision": "16-mixed", "warmup_steps": 300, "lr": 1e-4})
    top_train = OmegaConf.create({"precision": "16", "log_every_n_steps": 10})

    # Function signature: _merge_dataset_defaults(base_cfg, override_cfg)
    # where base_cfg = CLI/top-level, override_cfg = dataset defaults
    merged = _merge_dataset_defaults(top_train, dataset_train)
    # CLI (top_train) wins for precision
    assert merged.precision == "16"
    # Dataset defaults remain (shallow merge would incorrectly drop these)
    assert merged.log_every_n_steps == 10
    # Dataset-only key included
    assert merged.warmup_steps == 300
    assert merged.lr == 1e-4
