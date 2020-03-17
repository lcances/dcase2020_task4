"""
Not proper testing, but actual functional test
"""

from .src.datasetManager import DESEDManager

metadata_root="../dataset/DESED/dataset/metadata"
audio_root="../dataset/DESED/dataset/audio"

manager = DESEDManager(metadata_root, audio_root, 22050, validation_ratio=0.2, verbose=1)

print("Loading all subsets for training")
manager.add_subset("weak")
manager.add_subset("unlabel_in_domain")
manager.add_subset("synthetic20")

print("spliting 80 / 20 % train / validation")
manager.split_train_validation()

# TODO add some sanity check