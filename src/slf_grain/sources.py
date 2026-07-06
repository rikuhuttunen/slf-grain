import grain
import logging
import numpy as np
import pandas as pd
import sleeplab_format as slf

from pathlib import Path
from slf_grain.config import ComponentConfig, DatasetConfig
from typing import Any

logger = logging.getLogger(__name__)


class CropSource(grain.sources.RandomAccessDataSource):
    """A data source which outputs a cropped segment of each recording."""

    def __init__(self,
            ds_config: DatasetConfig,
            include_subject_ids: list[str] | None = None,
            seed: int | None = None) -> grain.MapDataset:
        self.ds_config = ds_config
        self.include_subject_ids = include_subject_ids
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Read the dataset directory tree
        # ds_tree = {
        #     self.ds_config.ds_dir.name: {
        #         series_dir.name: [
        #             subject_dir for subject_dir in series_dir.iterdir() if subject_dir.is_dir() and not subject_dir.name.startswith('.')
        #         ] for series_dir in self.ds_config.ds_dir.iterdir() if series_dir.is_dir() and not series_dir.name.startswith('.')
        #     }
        # }

        # Read the dataset subject directories to a list
        # TODO: logging not working?
        logger.info(f'Reading subject dirs from {ds_config.ds_dir / ds_config.series_name}')
        subject_dirs = self._get_subject_dirs(ds_config.ds_dir, ds_config.series_name)
        if include_subject_ids is not None:
            subject_dirs = [sdir for sdir in subject_dirs if sdir.name in include_subject_ids]

        logger.info(f'Parsing the ROI start and end times with type {ds_config.roi_src_type} and name {ds_config.roi_src_name}')
        self.subjects = [
            {'subject_dir': sdir, **dict(zip(
                ('roi_start', 'roi_end'),
                self._get_roi_start_end(sdir, ds_config.roi_src_type, ds_config.roi_src_name)
                ))
            }
            for sdir in subject_dirs
        ]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        return self._load_element(self.subjects[idx], self.ds_config, self._rng)

    @staticmethod
    def _get_subject_dirs(ds_dir: Path, series_name: str) -> list[Path]:
        series_dir = ds_dir / series_name
        subject_dirs = [
            subject_dir for subject_dir in series_dir.iterdir() if subject_dir.is_dir() and not subject_dir.name.startswith('.')
        ]
        
        return subject_dirs

    @staticmethod
    def _get_roi_start_end(subject_dir: Path, src_type: str, src_name: str, epoch_sec: float = 30.0) -> tuple[int, int]:
        """Resolve the region of interest based on roi_src_type and roi_src_name.""" 
        if src_type == 'annotation':
            annotations = CropSource._read_annotations(subject_dir, src_name)
            start = annotations[0].start_sec
            end = annotations[-1].start_sec + annotations[-1].duration
        elif src_type == 'sample_array':
            s, attributes = CropSource._read_sample_array(subject_dir, src_name, return_attributes=True)
            fs = attributes.sampling_rate
            slen = s.shape[0]
            start = 0.0
            end = float(slen / fs)
        elif src_type == 'lights_off_on':
            with open(subject_dir / 'metadata.json', 'rb') as f:
                raw_data = f.read()
                metadata = slf.models.SubjectMetadata.model_validate_json(raw_data)
            start = (metadata.lights_off - metadata.recording_start_ts).total_seconds()
            end = (metadata.lights_on - metadata.recording_start_ts).total_seconds()
        else:
            raise AttributeError(f'Unknown roi_src_type: {src_type}')
        
        # Calculate the signal length from first sample array
        sarr_dir = None
        for path in subject_dir.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                sarr_dir = path
                break
        
        s, attributes = CropSource._read_sample_array(subject_dir, sarr_dir, return_attributes=True)
        slen_sec = s.shape[0] // attributes.sampling_rate

        # If the last epoch is too short, discard it
        if end > slen_sec:
            end = end - epoch_sec
        # If the end time is still later than signal end, raise error
        assert end <= slen_sec, f'Subject {subject_dir.name}: signal length is {slen_sec}, tried to set end time to {end} s...'
        
        return start, end

    @staticmethod
    def _read_sample_array(subject_dir: Path, sarr_name: str, return_attributes: bool = False) -> np.array:
        s = np.load(subject_dir / sarr_name / 'data.npy', mmap_mode='r')
        if return_attributes:
            with open(subject_dir / sarr_name / 'attributes.json', 'rb') as f:
                raw_data = f.read()
                attributes = slf.models.ArrayAttributes.model_validate_json(raw_data)
                return s, attributes
        return s
    
    @staticmethod
    def _read_annotations(subject_dir: Path, annotation_name: str) -> list[slf.models.Annotation]:
        with open(subject_dir / f'{annotation_name}.a.json', 'rb') as f:
            raw_data = f.read()
            annotations = slf.models.BaseAnnotations.model_validate_json(raw_data)

        return annotations.annotations

    @staticmethod
    def _load_element(subject: dict[str, Any], ds_config: DatasetConfig, rng: np.random.Generator) -> dict[str, Any]:
        res = {}
        epoch_sec = 30.0

        if ds_config.start_sec == -1.0:
            start_interval = rng.uniform(
                subject['roi_start'] / epoch_sec,
                (subject['roi_end'] - ds_config.duration) / epoch_sec
            )
            start_sec = start_interval * epoch_sec
        else:
            start_sec = subject['roi_start'] + ds_config.start_sec

        if ds_config.duration == -1.0:
            # Load until the end if duration is -1
            duration = subject['roi_end'] - start_sec
        else:
            duration = ds_config.duration

        for k, cfg in ds_config.components.items():
            component = CropSource._load_component(subject['subject_dir'], cfg, start_sec, duration)
            if cfg.ctype == 'annotation' and cfg.return_type != 'segmentation_combined':
                component = {'values': component[0], 'labels': component[1]}
            
            res[k] = component

        return res

    @staticmethod
    def _load_component(subject_dir: Path, cfg: ComponentConfig, start_sec: float, duration: float) -> Any:
        if cfg.ctype == 'sample_array':
            return CropSource._load_sample_array(subject_dir, cfg, start_sec, duration)
        elif cfg.ctype == 'annotation':
            return CropSource._load_annotations(subject_dir, cfg, start_sec, duration)
        else:
            raise AttributeError(f'Unknown component type {cfg.ctype}')

    @staticmethod
    def _load_sample_array(subject_dir: Path, cfg: ComponentConfig, start_sec: float, duration: float) -> np.array:
        fs = cfg.fs
        start_idx = int(start_sec * fs)
        end_idx = int(start_idx + duration * fs)

        s = np.load(subject_dir / cfg.src_name / 'data.npy', mmap_mode='r')
        s = s[start_idx:end_idx]

        return s[..., np.newaxis]

    @staticmethod
    def _load_annotations(subject_dir: Path, cfg: ComponentConfig, frame_start_sec: float, frame_duration: float) -> Any:
        """TODO: Test if using pyarrow boosts performance"""
        annotations = CropSource._read_annotations(subject_dir, cfg.src_name)
        df = pd.DataFrame([dict(a) for a in annotations])

        if cfg.fs is not None:
            fs = cfg.fs
        else:
            fs = 1 / cfg.sampling_interval

        if len(df) > 0:
            # Calculate end times
            df['end_sec'] = df['start_sec'] + df['duration']

            # Filter out events outside the desired timeframe
            frame_end_sec = frame_start_sec + frame_duration
            df = df[df['end_sec'] > frame_start_sec]
            df = df[df['start_sec'] < frame_end_sec]

            # Filter the desired event types
            df = df[df['name'].isin(cfg.value_map.keys())]

            # Transform start and end times from seconds to indices wrt the timeframe
            df['start_idx'] = ((df['start_sec'] - frame_start_sec) * fs).astype(int)
            df['start_idx'] = df['start_idx'].clip(lower=0)

            df['end_idx'] = ((df['end_sec'] - frame_start_sec) * fs).astype(int)
            df['end_idx'] = df['end_idx'].clip(upper=int(fs*frame_duration))

        # Transform the dataframe to the desired output format
        if cfg.return_type == 'segmentation_combined':
            res = np.full(int(frame_duration * fs), cfg.value_map['_default'])
            for _, row in df.iterrows():
                res[row.start_idx:row.end_idx] = cfg.value_map[row['name']]

            return res

        elif cfg.return_type == 'segmentation_separate':
            res = {v: np.full(int(frame_duration * fs), cfg.value_map['_default']) for v in set(cfg.value_map.values())}
            for _, row in df.iterrows():
                res[cfg.value_map[row['name']]][row.start_idx:row.end_idx] = 1

            return res
        
        elif cfg.return_type == 'bbox':
            bboxes = []
            labels = []
            for _, row in df.iterrows():
                bboxes.append([row.start_idx, row.end_idx])
                labels.append(cfg.value_map[row['name']])

            return bboxes, labels

        else:
            raise AttributeError(f'Unsupported return_type {cfg.return_type}')
