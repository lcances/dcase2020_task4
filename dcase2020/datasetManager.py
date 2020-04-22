import os
import pandas
import tqdm
import h5py
import numpy as np
import pandas as pd
import librosa
import logging

import torch.utils.data

from .util.utils import feature_cache, multiprocess_feature_cache, timeit_logging

#log system
import logging
from dcase2020.util.log import log_flat
log = logging.getLogger(__name__)


class DatasetManager(object):
    LENGTH = 10
    
    cls_dict = {"Alarm_bell_ringing": 0, "Speech": 1, "Dog": 2,
                "Cat": 3, "Vacuum_cleaner": 4, "Dishes": 5,
                "Frying": 6, "Electric_shaver_toothbrush": 7,
                "Blender": 8,"Running_water": 9}

    cls_dict_reverse = dict(zip(cls_dict.values(),cls_dict.keys()))
    NB_CLASS = 10
    
    def __init__(self, metadata_root, audio_root, sampling_rate: int = 22050, verbose: int = 1):
        self.metadata_root = metadata_root
        self.audio_root = audio_root
        self.verbose = verbose
        self.sampling_rate = sampling_rate

        # use to check which subset are loaded
        self.loaded = {"weak": False, "unlabel_in_domain": False, "synthetic20": False}

        # recover dataset hdf file
        log.debug(os.path.join("..", "dataset", "dcase2020_dataset_%s.hdf5" % self.sampling_rate))
        self.hdf_dataset = os.path.join("..", "dataset", "dcase2020_dataset_%s.hdf5" % self.sampling_rate)

        # Prepare metadata container
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
        self._prepare_metadata()

        # verbose mode
        self.verbose = verbose
        if self.verbose == 1:
            self.tqdm_func = tqdm.tqdm
        elif self.verbose == 2:
            self.tqdm_func = tqdm.tqdm_notebook

    def _load_metadata(self):
        # load metadata for all training dataset
        for key in self.meta["train"]:
            path = os.path.join(self.metadata_root, "train", key + ".tsv")
            log.info("Reading metadata: %s" % path)

            df = pandas.read_csv(path, sep="\t")
            df.set_index("filename", inplace=True)

            self.meta["train"][key] = df
            
    def _prepare_metadata(self):
        """Change label string into binary vector, create weak truth using strong truth,
        rename column event_labels into strongID / weakID
        """
        def binarise(classes: str):
            """transform a list of string into a boolean vector"""
            output = [0] * DatasetManager.NB_CLASS
            if isinstance(classes, str):
                for cls in classes.split(","):
                    if cls != "":
                        output[DatasetManager.cls_dict[cls]] = 1

            return output
        
        def labelList_to_classID(df, column_label_name):
            # initialise a zero array
            zeros = [[0 for _ in range(len(DatasetManager.cls_dict))]] * len(df)
            df["classID"] = zeros
            
            # for each row, convert the list of event into weak binary vectors
            for name in df.index:
                df.at[name, "classID"] = binarise(df.at[name, column_label_name])

        def range_to_binaryarray(df, time_size):
            """Transform onset and offset into a binary vector of size <time_size>"""
            # intialise zero array
            zeros = [[0 for _ in range(time_size)]] * len(df)
            df["strongID"] = zeros

            # for each row, unroll the onset and offset into a binary vector
            for name in df.index:
                onset = df.at[name, "onset"]
                offset = df.at[name, "offset"]

                start = int(np.floor(onset * time_size / DatasetManager.LENGTH))
                end = int(np.ceil(offset * time_size / DatasetManager.LENGTH))

                df.at[name, "strongID"][start:end] = 1

        
        labelList_to_classID(self.meta["train"]["weak"], "event_labels")        # <-- "s" is not my doing :(

        labelList_to_classID(self.meta["train"]["synthetic20"], "event_label")
        range_to_binaryarray(self.meta["train"]["synthetic20"], self._feature_size())

    def get_subset(self, dataset: str, subset: str = None) -> dict:
        hdf_file = h5py.File(self.hdf_dataset, "r")

        if subset is not None:
            path = os.path.join(self.audio_root, dataset, subset)
        else:
            path = os.path.join(self.audio_root, dataset)

        output = self._hdf_to_dict(hdf_file, path)
        log.debug("output size: %s" % len(output))
        hdf_file.close()

        return output

    def _hdf_to_dict(self, hdf_file, path: str) -> dict:
        log.debug("hdf_file: %s" % hdf_file)
        log.debug("path: %s" % path)
        filenames = list(hdf_file[path]["filenames"])

        # read direct is much faster than looping over all the keys of the hdf group
        # ! Important while loading data from network drive !
        raw_audios = np.zeros(hdf_file[path]["data"].shape)
        hdf_file[path]["data"].read_direct(raw_audios)

        # minimun sanity check
        if len(filenames) != len(raw_audios):
            raise Warning("nb filenames != nb raw audio in subset %s" % path)

        output = dict(zip(filenames, raw_audios))

        return output

    @feature_cache
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

    def _feature_size(self) -> int:
        """Automatically compute the length of the mel-spectrogram using the sampling rate and default hop_length"""
        # TODO find a way to recover automatically the hop_length
        # Resolution scaling will change this value, leading to error while converting onset / offset into binary array
        return int(np.ceil(self.sampling_rate * DatasetManager.LENGTH / 512)) # <-- 512 = hop_length


class DESEDManager(DatasetManager):
    def __init__(self, metadata_root, audio_root, sampling_rate: int = 22050, verbose: int = 1,
            validation_ratio: float = 0.2):

        super().__init__(metadata_root, audio_root, sampling_rate, verbose)
        
        self.validation_ratio = validation_ratio
        self.validation_exist = False   # True if the function split_train_validation have been runned

        # prepare the variables
        self._X, self._y = dict() , None

        self.X = self._X        # if validation split not perform, then the default training set is all file loaded
        self.y = self._y
        self.X_val = dict()
        self.filenames = None

    def add_subset(self, key: str):
        """ Add a subset to the DESEDManager."""
        train_subsets = ["weak", "unlabel_in_domain", "synthetic20"]

        if key in train_subsets:
            dataset = "train"
            subset = key
        else:
            raise NotImplementedError("the subset %s is not suppoerted yet" % key)

        logging.info("Loading dataset: %s, subset: %s" % (dataset, subset))

        # Load the audio and concat directly in main dict self._X
        self._X = {**self._X, **self.get_subset(dataset, subset)}
        
        # concat the metadata into self._y
        target_meta = self.meta[dataset][subset] if key in train_subsets else self.meta[dataset]
        self._y = pd.concat([self._y, target_meta]) # TODO check if need to specify an axis

        # mark the subset as loaded
        self.loaded[key] = True

    def split_train_validation(self):
        if self.validation_exist:
            log.info("Reverting previous split")
            self.reset_split()

        log.info("Creating new train / validation split")
        log.info("validation ratio : %s" % self.validation_ratio)
        
        # TODO check if there is not a more efficient way, maybe a .select with drop
        filenames = list(self._X.keys())
        self.X, self.y = dict(), dict()

        nb_file = len(filenames)
        nb_validation = int(nb_file * self.validation_ratio)

        # Random selection of the validation fold --- TODO add a balance split function
        validation_filenames = np.random.choice(filenames, size=nb_validation)

        # split the audio dictionary into a training and validation one (X, X_val)
        # deleting file directly after the move to avoid high memory usage
        for name in filenames:
            if name not in validation_filenames:
                self.X[name] = self._X[name]
            else:
                self.X_val[name] = self._X[name]

            self._X.pop(name, None)

        # split the metadata dataframe into a training and validation ones (y, y_val)
        # TODO check if there is not a more efficient way, maybe a .select with drop
        self.filenames = list(self.X.keys())
        
        self.y_val = self._y.loc[self._y.index.isin(validation_filenames)]
        self.y = self._y.loc[self._y.index.isin(self.filenames)]
        
        # set the validation flag to allow datasets to use validation set
        self.validation_exist = True

    def reset_split(self): 
        # To avoid reloading the data but perform a new split
        # concat X and X_val into _X
        if self.validation_exist:
            self._X = {**self.X, **self.X_val}
            self.filenames = None
 


        



if __name__ == '__main__':
    metadata_root="../dataset/DESED/dataset/metadata"
    audio_root="../dataset/DESED/dataset/audio"

    manager = DESEDManager(metadata_root, audio_root, 22050, validation_ratio=0.2, verbose=1)

    print("")
    manager.add_subset("weak")
    manager.add_subset("unlabel_in_domain")
    manager.add_subset("synthetic20")
    manager.split_train_validation()

