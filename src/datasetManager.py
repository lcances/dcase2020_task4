import os
import pandas

import torch.utils.data


class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, metadata_root, audio_root):
        self.metadata_root = metadata_root
        self.audio_root = audio_root

        self.meta = {
            "train": {
                "weak": None,
                "unlabel_in_domain": None,
                "synthetic20": None,
            },

            "validation": None,
            "evaluation": None,
        }

        self.audio = {
            "train": {
                "weak": None,
                "unlabel_in_domain": None,
                "synthetic20": None,
            },

            "validaiton": None,
            "evaluation": None,
        }

        self._load_metadata()

    def _load_metadata(self):
        # load metadata for all training dataset
        for key in self.meta["train"]:
            path = os.path.join(self.metadata_root, "train", key + ".tsv")

            self.meta["train"][key] = pandas.read_csv(path, sep="\t")

    def _hdf_to_dict(self):
        pass


if __name__ == '__main__':
    manager = DatasetManager(
        metadata_root="../dataset/DESED/dataset/metadata",
        audio_root="../dataset/DESED/dataset/audio"
    )

    print(manager.meta["train"])
