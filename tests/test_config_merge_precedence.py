from omegaconf import OmegaConf


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

    merged_model = OmegaConf.merge(cfg.dataset.model, cfg.model)
    merged_train = OmegaConf.merge(cfg.dataset.train, cfg.train)

    # Top-level values (including CLI overrides) must win.
    assert merged_model.d_model == 128
    assert merged_model.dropout == 0.1
    assert merged_train.precision == "32"
    assert merged_train.lr == 1e-3
