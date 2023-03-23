import logging
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as skshuffle
from torch.utils.data import DataLoader
from src.pytorch.utils.helpers import get_memory_usage_mb
import src.pytorch.utils.default_args as default_args

from src.pytorch.training_data import (
    InstanceDataset,
    load_training_state_value_pairs,
)

_log = logging.getLogger(__name__)


class KFoldTrainingData:
    def __init__(
        self,
        samples_file: str,
        device: torch.device,
        max_training_memory: int = default_args.TRAIN_MAX_MEMORY,
        batch_size: int = default_args.TRAIN_BATCH_SIZE,
        num_folds: int = default_args.TRAIN_NUM_FOLDS,
        output_layer: str = default_args.TRAIN_OUTPUT_LAYER,
        shuffle: bool = default_args.TRAIN_SHUFFLE,
        seed: int = default_args.DEFAULT_SEED,
        shuffle_seed: int = default_args.DEFAULT_SEED,
        training_size: float = default_args.TRAIN_TRAINING_SIZE,
        data_num_workers: int = 0,
        normalize: bool = default_args.TRAIN_NORMALIZE_OUTPUT,
        model: str = default_args.TRAIN_MODEL,
    ):
        assert training_size > 0.0 and training_size <= 1.0

        self.device = device

        _log.info("Reading and preparing data...")
        (
            self.states,
            self.heuristics,
            self.sample_max_value,
        ) = load_training_state_value_pairs(samples_file, max_training_memory)
        _log.info(f"Number of samples: {len(self.states)}")
        _log.info(f"Mem usage after loading data: {get_memory_usage_mb()} MB")

        self.normalize = normalize
        if self.normalize:
            for i in range(len(self.heuristics)):
                self.heuristics[i] /= self.sample_max_value
        self.batch_size = batch_size if batch_size > 0 else None
        self.num_folds = num_folds
        self.output_layer = output_layer
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_seed = shuffle_seed
        self.training_size = training_size
        self.data_num_workers = data_num_workers
        self.model = model
        self.kfolds = self.generate_kfold_training_data()

    def generate_kfold_training_data(self) -> list:
        """
        Generates the folds.
        Returns two list of tuples of size num_folds: dataloaders and problems.
        The first item corresponds to train set, the second to val set, and the
        third to test set.
        """
        _log.info(f"Generating {self.num_folds}-fold...")

        kfolds = []
        instances_per_fold = int(len(self.states) / self.num_folds)
        for i in range(self.num_folds):
            x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []
            if self.num_folds == 1:
                if self.training_size == 1.0:
                    x_train, y_train = (
                        skshuffle(
                            self.states,
                            self.heuristics,
                            random_state=self.shuffle_seed,
                        )
                        if self.shuffle
                        else (self.states, self.heuristics)
                    )

                else:
                    x_train, x_val, y_train, y_val = train_test_split(
                        self.states,
                        self.heuristics,
                        train_size=self.training_size,
                        shuffle=self.shuffle,
                        random_state=self.shuffle_seed,
                    )

            else:
                for j in range(len(self.states)):
                    if int(j / instances_per_fold) == i:
                        x_test.append(self.states[j])
                        y_test.append(self.heuristics[j])
                    else:
                        x_train.append(self.states[j])
                        y_train.append(self.heuristics[j])

            self.states = None
            self.heuristics = None

            worker_fn = (
                None
                if self.seed == -1
                else lambda id: np.random.seed(self.shuffle_seed % 2**32)
            )

            g = None if self.seed == -1 else torch.Generator()
            if g is not None:
                g.manual_seed(self.shuffle_seed)

            pin_mem = True if self.device == torch.device("cuda:0") else False

            _log.debug(f"Mem usage before train_dataloader: {get_memory_usage_mb()} MB")
            train_dataloader = DataLoader(
                dataset=InstanceDataset(
                    x_train, y_train, self.sample_max_value, self.output_layer
                ),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.data_num_workers,
                worker_init_fn=worker_fn,
                generator=g,
                pin_memory=pin_mem,
            )
            _log.info(f"Created train dataloader.")
            _log.debug(f"Mem usage: {get_memory_usage_mb()} MB")

            x_train = None
            y_train = None

            val_dataloader = (
                DataLoader(
                    dataset=InstanceDataset(
                        x_val, y_val, self.sample_max_value, self.output_layer
                    ),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.data_num_workers,
                    worker_init_fn=worker_fn,
                    generator=g,
                    pin_memory=pin_mem,
                )
                if len(x_val) != 0
                else None
            )
            _log.info(f"Created validation dataloader.")
            _log.debug(f"Mem usage: {get_memory_usage_mb()} MB")

            x_val = None
            y_val = None

            test_dataloader = (
                DataLoader(
                    dataset=InstanceDataset(
                        x_test, y_test, self.sample_max_value, self.output_layer
                    ),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.data_num_workers,
                    worker_init_fn=worker_fn,
                    generator=g,
                    pin_memory=pin_mem,
                )
                if len(x_test) != 0
                else None
            )
            _log.info(f"Created test dataloader.")
            _log.debug(f"Mem usage: {get_memory_usage_mb()} MB")

            x_test = None
            y_test = None

            kfolds.append((train_dataloader, val_dataloader, test_dataloader))
            _log.info(f"Mem usage after creating fold(s): {get_memory_usage_mb()} MB")

        return kfolds

    def get_fold(self, idx: int) -> tuple:
        """
        Returns a fold as tuple(train dataloader, test dataloader).
        Counting from 0.
        """
        return self.kfolds[idx]
