import collections
import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd

from datasetManager import DatasetManager, DESEDManager
import torch.utils.data

from augmentation_utils.signal_augmentations import SignalAugmentation
from augmentation_utils.spec_augmentations import SpecAugmentation

import logging

class DESEDDataset(torch.utils.data.Dataset):
    def __init__(self, manager: DatasetManager, train: bool, val: bool, augments=(), cached=False):
        self.manager = manager
        self.train = train
        self.val = val
        self.augments = augments
        self.cached = cached

        if len(augments) != 0 and cached:
            logging.info("Cache system deactivate due to usage of online augmentation")
            self.cached = False

        self._check_arguments()

        if self.train:
            self.X = self.manager.X
            self.y = self.manager.y
        elif self.val:
            self.X = self.manager.X_val
            self.y = self.manager.y_val

        self.filenames = list(self.X.keys())

        # alias for verbose mode
        self.tqdm_func = self.manager.tqdm_func

    def _check_arguments(self):
        if sum([self.train, self.val]) != 1:
            raise AssertionError("Train and val and mutually exclusive")
        
    def __len__(self):
        nb_file = len(self.filenames)
        return nb_file

    def __getitem__(self, index):
        filename = self.filenames[index]
        return self._generate_data(filename)

    def _generate_data(self, filename: str):
        # load the raw_audio
        raw_audio = self.X[filename]

        # recover ground truth
        y = self.y.at[filename, "classID"]

        raw_audio = self._apply_augmentation(raw_audio, SignalAugmentation)
        raw_audio = self._pad_and_crop(raw_audio)

        # extract feature and apply spec augmentation
        feat = self.manager.extract_feature(raw_audio, filename=filename, cached=self.cached)
        feat = self._apply_augmentation(feat, SpecAugmentation)
        y = np.asarray(y)

        return feat, y

    def _pad_and_crop(self, raw_audio):
        LENGTH = DESEDManager.LENGTH
        SR = self.manager.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio

    def _apply_augmentation(self, data, augType):
        np.random.shuffle(self.augments)
        for augment_func in self.augments:
            if isinstance(augment_func, augType):
                return augment_func(data)

        return data