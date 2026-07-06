import pytest

from pathlib import Path
from sleeplab_format.test_utils.fixtures import dataset, series, subjects, subject_ids
from sleeplab_format import writer


@pytest.fixture(scope='session')
def example_config_path():
    data_dir = Path(__file__).parent / 'data'
    return data_dir / 'example_config.yml'


@pytest.fixture
def ds_dir(dataset, tmp_path):
    basedir = tmp_path / 'datasets'
    writer.write_dataset(dataset, basedir, annotation_format='json')

    return basedir / dataset.name
