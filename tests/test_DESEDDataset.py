import json
import os.path as osp

from argparse import ArgumentParser, Namespace

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, default="../dataset/DESED/")
	return parser.parse_args()


def test():
	args = create_args()
	desed_metadata_root = osp.join(args.dataset, "dataset", "metadata")
	desed_audio_root = osp.join(args.dataset, "dataset", "audio")

	manager = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=True,
		sampling_rate=22050,
		verbose=1,
	)

	# manager.add_subset("weak")
	manager.add_subset("synthetic20")
	# manager.add_subset("unlabel_in_domain")

	ds = DESEDDataset(manager, train=True, val=False, augments=[], cached=False, weak=True, strong=True)

	print("len : ", len(ds))  # weak = 11808, synthetic20 = 2584

	print("Strong sizes : ")
	x, y = ds[0]
	print("x = ", x.shape)  # (64, 431)
	print("y[0] = ", y[0].shape)  # (10,)
	print("y[1] = ", y[1].shape)  # (10, 431)

	data = {"x": x.tolist(), "y_weak": y[0].tolist(), "y_strong": y[1].tolist()}
	with open("spec.json", "w") as file:
		json.dump(data, file)


if __name__ == "__main__":
	test()
