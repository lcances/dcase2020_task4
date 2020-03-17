import os
import pandas
import tqdm
import h5py
import numpy as np
import librosa

import torch.utils.data

from utils import feature_cache, multiprocess_feature_cache


class DatasetManager(torch.utils.data.Dataset):
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

        self.audio = {
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

    def _load_audio(self):
        # load raw audio for all training set
        for key in self.meta["train"]:
            self.load_subset("train", key)

    def load_subset(self, dataset: str, subset: str = None):
        hdf_file = h5py.File(self.hdf_dataset, "r")

        if subset is not None:
            path = os.path.join(self.audio_root, dataset, subset)
            self.audio[dataset][subset] = self._hdf_to_dict(hdf_file, path)
        else:
            path = os.path.join(self.audio_root, dataset)
            self.audio[dataset] = self._hdf_to_dict(hdf_file, path)

        hdf_file.close()

    def _hdf_to_dict(self, hdf_file, path: str):
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
            subset: str = "weak", validation: bool = False, validation_ratio: float = 0.2):

        super().__init__(metadata_root, audio_root, sampling_rate, verbose)
        
        self.subset = subset
        self.validation_ratio = validation_ratio

        # load only the concern audio
        self.load_subset("train", self.subset)

        # prepare the variables
        self.X = self.audio["train"][self.subset]
        self.y = self.audio["train"][self.subset]

        self.filenames = self.X.keys()

        # if it is a validation set
        self._prepare_validation()

    def _prepare_validation(self):
        # count how many file is in the validation fold
        nb_file = len(self.filenames)
        nb_validation = nb_file // self.validation_ratio

        # pick it from the filename list
        validation_filenames = np.random.choice(nb_validation, size=nb_validation)

        # remove other file from the dictionnary to free memory
        for name in self.X:
            if name not in validation_filenames:
                self.X.pop(name, None)

        # remove other file from the metadata
        for name in self.y.filename.value:
            if name not in validation_filenames:
                self.y.drop(name, inplace=True)

        



if __name__ == '__main__':
    metadata_root="../dataset/DESED/dataset/metadata"
    audio_root="../dataset/DESED/dataset/audio"

    train_weak_dataset = DESEDManager(metadata_root, audio_root, 22050, subset="weak", validation=False)
    val_weak_dataset = DESEDManager(metadata_root, audio_root, 22050, subset="weak", validation=True, validation_ratio=0.2)

    train_uid_dataset = DESEDManager(metadata_root, audio_root, 22050, subset="unlabel_in_domain", validation=False)
    val_uid_dataset = DESEDManager(metadata_root, audio_root, 22050, subset="unlabel_in_domain", validation=True, validation_ratio=0.2)

    train_synth_dataset = DESEDManager(metadata_root, audio_root, 22050, subset="synthetic20", validation=False)
    val_synth_dataset = DESEDManager(metadata_root, audio_root, 22050, subset="synthetic20", validation=True, validation_ratio=0.2)

