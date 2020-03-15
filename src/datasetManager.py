import os
import pandas
import tqdm
import h5py
import numpy as np
import librosa

import torch.utils.data

from .utils import feature_cache, multiprocess_feature_cache


class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, metadata_root, audio_root, sampling_rate: int = 22050, preload: bool = True, verbose: int = 1):
        self.metadata_root = metadata_root
        self.audio_root = audio_root
        self.preload = preload
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
        self._load_audio()

    def _load_metadata(self):
        # load metadata for all training dataset
        for key in self.meta["train"]:
            path = os.path.join(self.metadata_root, "train", key + ".tsv")
            print(path)

            df = pandas.read_csv(path, sep="\t")
            df.set_index("filename", inplace=True)

            self.meta["train"][key] = df

    def _load_audio(self):
        hdf_file = h5py.File(self.hdf_dataset, "r")

        # load raw audio for all training set
        for key in self.meta["train"]:
            self.load_subset(hdf_file, "train", key)

        hdf_file.close()

    def load_subset(self, hdf_file, dataset: str, subset: str = None):
        if subset is not None:
            path = os.path.join(self.audio_root, dataset, subset)
            self.audio[dataset][subset] = self._hdf_to_dict(hdf_file, path)
        else:
            path = os.path.join(self.audio_root, dataset)
            self.audio[dataset] = self._hdf_to_dict(hdf_file, path)

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

if __name__ == '__main__':
    manager = DatasetManager(
        metadata_root="../dataset/DESED/dataset/metadata",
        audio_root="../dataset/DESED/dataset/audio"
    )

    print(manager.meta["train"])
