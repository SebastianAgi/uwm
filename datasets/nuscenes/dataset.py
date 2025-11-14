import copy
import math
import pathlib

import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.normalizer import LinearNormalizer, NestedDictLinearNormalizer
from datasets.utils.obs_utils import unflatten_obs
from datasets.utils.sampler import TrajectorySampler

class NusceneDataset(Dataset):
    def __init__(
        self,
        name: str,
        buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        history_len: int = 1,
        normalize_lowdim: bool = False,
        normalize_action: bool = False,
        val_ratio: float = 0.0,
        num_workers: int = 8,
    ):
        self.name = name
        self.seq_len = seq_len
        self.history_len = history_len
        self.num_workers = num_workers

        # Parse observation and action shapes
        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self._action_shape = tuple(shape_meta["action"]["shape"])

        # Compressed buffer to store episode data
        self.buffer_dir = pathlib.Path(buffer_path).parent
        self.buffer = self._init_buffer(buffer_path)

        # Create training-validation split
        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)