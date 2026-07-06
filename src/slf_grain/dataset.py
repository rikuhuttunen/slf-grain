import grain
import logging
import numpy as np
import slf_grain

from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def from_slf(
        ds_config: slf_grain.config.DatasetConfig,
        include_subject_ids: list[str],
        seed: int | None = None) -> grain.MapDataset:
    if ds_config.start_sec == -1.0:
        assert ds_config.duration > 0.0, 'Need to define start_sec or duration'

    source = slf_grain.sources.CropSource(ds_config, include_subject_ids, seed)
    return grain.MapDataset.source(source).seed(seed)


def parse_split_configs(cfg: dict[str, Any]) -> dict[str, slf_grain.config.DatasetConfig]:
    """Parse configs for datasets and splits from dicts to DatasetConfigs."""
    cfg = cfg.copy()  # Copy so that the original config does not get modified
    splits = cfg.pop('splits')
    res = {}

    for k, split in splits.items():
        _cfg_dict = cfg.copy()
        _cfg_dict.update(split['config'])
        _cfg = slf_grain.config.DatasetConfig.model_validate(_cfg_dict)
        res[k] = _cfg

    return res


def load_split(cfg: dict[str, Any], seed: int = None) -> dict[str, grain.MapDataset]:
    logger.info(f'Reading subject IDs for dataset {cfg["ds_dir"]}, series {cfg["series_name"]}')
    subject_dirs = slf_grain.sources.CropSource._get_subject_dirs(Path(cfg['ds_dir']), cfg['series_name'])
    subj_ids = [sdir.name for sdir in subject_dirs]
    
    logger.info('Creating the splits...')
    # Assert that the sum of split sizes matches the total number of subjects
    sum_split_sizes = sum([split['n'] for split in cfg['splits'].values()])
    _msg = f'Sum of split sizes does not match dataset size ({sum_split_sizes} != {len(subj_ids)})'
    assert sum_split_sizes == len(subj_ids), _msg

    # Randomly shuffle before splitting
    rng = np.random.default_rng(seed)
    permuted_subj_ids = rng.permutation(subj_ids)

    # Split the subject IDs
    split_subj_ids = {}
    curr_split_start = 0
    for k, split in cfg['splits'].items():
        curr_split_end = curr_split_start + split['n']
        split_subj_ids[k] = permuted_subj_ids[curr_split_start:curr_split_end]
        curr_split_start = curr_split_end

    # Parse config for each split
    split_cfgs = parse_split_configs(cfg)
    
    # Create the datasets for each split
    res = {}
    for k, _cfg in split_cfgs.items():
        # TODO: is it OK to use same seed for all subsets?
        res[k] = from_slf(
            _cfg, include_subject_ids=split_subj_ids[k], seed=seed)
    
    return res


def load_split_concat(cfgs: dict[str, Any], seed: int = 42) -> dict[str, grain.MapDataset]:
    """Load, split, and concatenate multiple sleeplab-format datasets.
    
    Args:
        cfgs: A dict with entries for each dataset.
        seed: The seed for random splitting.
    
    Returns:
        A dict of grain MapDatasets with entries for each split.
    """
    ds_list = [load_split(cfgs[k], seed=seed) for k in cfgs.keys()]
    split_keys = set([k for ds in ds_list for k in ds.keys()])
    datasets = {k: grain.MapDataset.concatenate([ds[k] for ds in ds_list if k in ds.keys()]) for k in split_keys}

    return datasets
