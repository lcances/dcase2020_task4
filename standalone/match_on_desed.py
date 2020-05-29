import os.path as osp

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader

from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset
from dcase2020_task4.util.utils import reset_seed


def get_desed_loaders(desed_metadata_root: str, desed_audio_root: str) -> (DataLoader, DataLoader):
	manager = DESEDManager(
		desed_metadata_root, desed_audio_root,
		sampling_rate=22050,
		validation_ratio=0.2,
		verbose=1
	)

	manager.add_subset("weak")
	manager.add_subset("unlabel_in_domain")
	manager.split_train_validation()

	train_dataset = DESEDDataset(manager, train=True, val=False, augments=[], cached=True)
	val_dataset = DESEDDataset(manager, train=False, val=True, augments=[], cached=True)

	batch_size = 16

	# loaders
	training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	return training_loader, val_loader


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, default="../dataset/DESED")
	parser.add_argument("--seed", type=int, default=123)
	return parser.parse_args()


def main():
	args = create_args()

	reset_seed(args.seed)

	desed_metadata_root = osp.join(args.dataset, "dataset/metadata")
	desed_audio_root = osp.join(args.dataset, "dataset/audio")

	loader_train, loader_val = get_desed_loaders(desed_metadata_root, desed_audio_root)

	for i, data in enumerate(loader_train):
		print(data)
		if i > 5:
			break


if __name__ == "__main__":
	main()
