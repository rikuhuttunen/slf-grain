from slf_grain import config, sources


def test_load_source(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    cfg.ds_dir = ds_dir

    source = sources.CropSource(cfg)

    assert len(source.subjects) == 3
