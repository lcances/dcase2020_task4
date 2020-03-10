import torch

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, metadata_root, audio_root):
        self.metadata_root = metadata_root
        self.audio_root = audio_root

    def _load_metadata(self):
        pass

    def _hdf_to_dict(self):
        pass