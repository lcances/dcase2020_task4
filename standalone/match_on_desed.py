import numpy as np
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice, Compose

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

from dcase2020_task4.baseline.models import WeakBaseline
from dcase2020_task4.train_fixmatch import train_fixmatch, default_fixmatch_hparams
from dcase2020_task4.util.NoLabelDataLoader import NoLabelDataLoader
from dcase2020_task4.util.other_metrics import CategoricalConfidenceAccuracy, FnMetric
from dcase2020_task4.util.rgb_augmentations import RandCrop
from dcase2020_task4.util.utils import reset_seed, get_datetime
from dcase2020_task4.util.utils_match import to_batch_fn


def create_args() -> Namespace:
	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf", "sp"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--dataset", type=str, default="../dataset/DESED")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="WeakBaseline", choices=["WeakBaseline"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.5)
	parser.add_argument("--mode", type=str, default="multihot")
	return parser.parse_args()


def get_desed_loaders(args) -> (DataLoader, DataLoader, DataLoader):
	desed_metadata_root = osp.join(args.dataset, osp.join("dataset", "metadata"))
	desed_audio_root = osp.join(args.dataset, osp.join("dataset", "audio"))

	manager_s = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=True,
		sampling_rate=22050,
		validation_ratio=0.2,
		verbose=1
	)
	manager_s.add_subset("weak")
	manager_s.split_train_validation()

	manager_u = DESEDManager(
		desed_metadata_root, desed_audio_root,
		from_disk=True,
		sampling_rate=22050,
		validation_ratio=0.0,
		verbose=1
	)
	manager_u.add_subset("unlabel_in_domain")
	manager_u.split_train_validation()

	ds_train_s = DESEDDataset(manager_s, train=True, val=False, augments=[], cached=True, weak=True, strong=False)
	ds_val = DESEDDataset(manager_s, train=False, val=True, augments=[], cached=True, weak=True, strong=False)
	ds_train_u = DESEDDataset(manager_u, train=True, val=False, augments=[], cached=True, weak=False, strong=False)

	loader_train_s = DataLoader(ds_train_s, batch_size=args.batch_size, shuffle=True)
	loader_train_u = NoLabelDataLoader(ds_train_u, batch_size=args.batch_size, shuffle=True)
	loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

	return loader_train_s, loader_train_u, loader_val


def main():
	args = create_args()

	reset_seed(args.seed)

	hparams = edict()
	hparams.update(args.__dict__)
	hparams.begin_date = get_datetime()

	model_factory = lambda: WeakBaseline()
	acti_fn = lambda batch, dim: batch.sigmoid()

	weak_augm_fn = to_batch_fn(RandomChoice([
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	]))
	strong_augm_fn = to_batch_fn(Compose([
		Transform(0.75, scale=(0.5, 1.5)),
		Transform(0.75, rotation=(-np.pi, np.pi)),
		RandCrop(1.0),
	]))
	metrics_s = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_val_lst = [CategoricalConfidenceAccuracy(hparams.confidence), FnMetric(binary_cross_entropy)]
	metrics_val_names = ["acc", "loss"]

	pre_batch_fn = lambda batch: torch.as_tensor(batch).cuda().float()
	pre_labels_fn = lambda label: torch.as_tensor(label).cuda().float()

	loader_train_s, loader_train_u, loader_val = get_desed_loaders(args)

	hparams_fm = default_fixmatch_hparams()
	hparams_fm.update(hparams)
	train_fixmatch(
		model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, weak_augm_fn, strong_augm_fn,
		metrics_s, metrics_u, metrics_val_lst, metrics_val_names, pre_batch_fn, pre_labels_fn, hparams_fm
	)


if __name__ == "__main__":
	main()
