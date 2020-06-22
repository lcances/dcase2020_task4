"""
Not proper testing, but actual functional test
"""

from dcase2020.datasetManager import DESEDManager

metadata_root="../dataset/DESED/dataset/metadata"
audio_root="../dataset/DESED/dataset/audio"

manager = DESEDManager(metadata_root, audio_root, 22050, from_disk=True, verbose=1)

print("Loading all subsets for training")
manager.add_subset("weak")
#manager.add_subset("unlabel_in_domain")
#manager.add_subset("synthetic20")

# TODO add some sanity check