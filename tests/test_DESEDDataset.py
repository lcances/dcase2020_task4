import os.path as osp
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset


def test():
	dataset = "../dataset/DESED"
	desed_metadata_root = osp.join(dataset, osp.join("dataset", "metadata"))
	desed_audio_root = osp.join(dataset, osp.join("dataset", "audio"))

	manager = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=True,
		sampling_rate=22050,
		validation_ratio=0.2,
		verbose=1
	)

	# manager.add_subset("weak")
	manager.add_subset("synthetic20")
	manager.split_train_validation()

	ds = DESEDDataset(manager, train=False, val=True, augments=[], cached=False, weak=False, strong=True)

	x, y = ds[0]
	print(x.shape)
	print(y[0].shape)


if __name__ == "__main__":
	test()
