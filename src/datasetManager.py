import os
import pandas
import tqdm
import h5py
import numpy as np
import pandas as pd
import librosa
import logging

import torch.utils.data

from utils import feature_cache, multiprocess_feature_cache, timeit_logging


class DatasetManager:
    def __init__(self, metadata_root, audio_root, sampling_rate: int = 22050, verbose: int = 1):
        self.metadata_root = metadata_root
        self.audio_root = audio_root
        self.verbose = verbose
        self.sampling_rate = sampling_rate

        # recover dataset hdf file
        self.hdf_dataset = os.path.join("..", "dataset", "dcase2020_dataset_%s.hdf5" % self.sampling_rate)

        # verbose mode
        self.verbose = verbose
        if self.verbose == 1:
            self.tqdm_func = tqdm.tqdm
        elif self.verbose == 2:
            self.tqdm_func = tqdm.tqdm_notebook

        self.meta = {
            "train": {
                "weak": {},
                "unlabel_in_domain": {},
                "synthetic20": {},
            },

            "validation": {},
            "evaluation": {},
        }

        self._load_metadata()

    def _load_metadata(self):
        # load metadata for all training dataset
        for key in self.meta["train"]:
            path = os.path.join(self.metadata_root, "train", key + ".tsv")
            print(path)

            df = pandas.read_csv(path, sep="\t")
            df.set_index("filename", inplace=True)

            self.meta["train"][key] = df

    def get_subset(self, dataset: str, subset: str = None) -> dict:
        hdf_file = h5py.File(self.hdf_dataset, "r")

        if subset is not None:
            path = os.path.join(self.audio_root, dataset, subset)
        else:
            path = os.path.join(self.audio_root, dataset)

        hdf_file.close()

        return self._hdf_to_dict(hdf_file, path)

    def _hdf_to_dict(self, hdf_file, path: str) -> dict:
        print(path)
        filenames = list(hdf_file[path]["filenames"])

        raw_audios = np.zeros(hdf_file[path]["data"].shape)
        hdf_file[path]["data"].read_direct(raw_audios)

        # minimun sanity check
        if len(filenames) != len(raw_audios):
            raise Warning("nb filenames != nb raw audio in subset %s" % path)

        output = dict(zip(filenames, raw_audios))

        return output

    @multiprocess_feature_cache
    def extract_feature(self, raw_data, filename = None, cached = False):
        """
        extract the feature for the model. Cache behaviour is implemented with the two parameters filename and cached
        :param raw_data: to audio to transform
        :param filename: the key used by the cache system
        :param cached: use or not the cache system
        :return: the feature extracted from the raw audio
        """
        feat = librosa.feature.melspectrogram(
            raw_data, self.sampling_rate, n_fft=2048, hop_length=512, n_mels=64, fmin=0, fmax=self.sampling_rate // 2)
        feat = librosa.power_to_db(feat, ref=np.max)
        return feat


class DESEDManager(DatasetManager):
    def __init__(self, metadata_root, audio_root, sampling_rate: int = 22050, verbose: int = 1,
            validation_ratio: float = 0.2):

        super().__init__(metadata_root, audio_root, sampling_rate, verbose)
        
        self.validation_ratio = validation_ratio
        self.validation_exist = False   # True if the function split_train_validation have been runned

        # prepare the variables
        self._X, self._y = None , None

        self.X = self._X        # if validation split not perform, then the default training set is all file loaded
        self.y = self._y
        self.val_X = None
        self.filenames = None

    @timeit_logging
    def add_subset(self, key: str):
        train_subsets = ["weak", "unlabel_in_domain", "synthetic20"]

        if key in train_subsets:
            dataset = "train"
            subset = key
        else:
            dataset = key
            subset = None

        logging.info("Loading dataset: %s, subset: %s" % (dataset, subset))

        # Load the audio and concat directly in main dict self._X
        self._X = {**self._X, **self.get_subset(dataset, subset)}
        
        # concat the metadata into self._y
        target_meta = self.meta[dataset][subset] if key in train_subsets else self.meta[dataset]
        self._y = pd.concat([self._y, target_meta]) # TODO check if need to specify an axis

    def split_train_validation(self):
        logging.warning("The function consume the previous load of the data. In order to perform a new split, \
        data must be reset and reloaded")

        # TODO check if there is not a more efficient way, maybe a .select with drop
        filenames = list(self._X.keys())

        self.X, self.y = dict(), dict()

        # count how many file is in the validation fold
        nb_file = len(filenames)
        nb_validation = nb_file // self.validation_ratio

        # pick it from the filename list
        validation_filenames = np.random.choice(nb_validation, size=nb_validation)

        # split the audio dictionary into a training and validation one (X, X_val)
        # deleting file directly after the move to avoid high memory usage
        for name in self._X:
            if name not in validation_filenames:
                self.X[name] = self._X[name]
            else:
                self.X_val = self._X[name]

            self._X.pop(name, None)

        # set the filenames
        self.filenames = list(self.X.keys())

        # split the metadata dataframe into a training and validation ones (y, y_val)
        # TODO check if there is not a more efficient way, maybe a .select with drop
        self.y = self._y.loc[self._y.index.isin(validation_filenames)]
        self.y_val = self._y.loc[self._y.index.isin(self.filenames)]

        # set the validation flag to allow datasets to use validation set
        self.validation_exist

    def reset(self): 
        # TODO check if it actually work
        # all the audio and metadata are delete
        self._X, self._y = None , None

        self.X = self._X
        self.y = self._y
        self.val_X = None
        self.filenames = None
 


        



if __name__ == '__main__':
    metadata_root="../dataset/DESED/dataset/metadata"
    audio_root="../dataset/DESED/dataset/audio"

    manager = DESEDManager(metadata_root, audio_root, 22050, validation_ratio=0.2, verbose=1)

    manager.add_subset("weak")
    manager.add_subset("unlabel_in_domain")
    manager.add_subset("synthetic20")
    manager.split_train_validation()

