from slf_grain import config


def test_parse_config(example_config_path):
    cfg = config.parse_config(example_config_path)
    assert type(cfg) == config.DatasetConfig