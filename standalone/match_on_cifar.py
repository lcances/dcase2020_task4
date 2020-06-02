import numpy as np
import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomChoice, Compose

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip
from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.util.FnDataLoader import FnDataLoader
from dcase2020_task4.util.NoLabelDataLoader import NoLabelDataLoader
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor
from dcase2020_task4.util.other_metrics import CategoricalConfidenceAccuracy, MaxMetric, FnMetric
from dcase2020_task4.util.utils_match import cross_entropy, to_batch_fn

from dcase2020_task4.resnet import ResNet18
from dcase2020_task4.train_fixmatch import train_fixmatch, default_fixmatch_hparams
from dcase2020_task4.train_mixmatch import train_mixmatch, default_mixmatch_hparams
from dcase2020_task4.train_remixmatch import train_remixmatch, default_remixmatch_hparams
from dcase2020_task4.train_supervised import train_supervised, default_supervised_hparams
from dcase2020_task4.vgg import VGG


def create_args() -> Namespace:
	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf", "sp"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--dataset", type=str, default="../../dataset/CIFAR10")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--model_name", type=str, default="VGG11", choices=["VGG11", "ResNet18"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--dataset_ratio", type=float, default=1.0)
	parser.add_argument("--supervised_ratio", type=float, default=0.1)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.3)
	parser.add_argument("--mode", type=str, default="onehot")
	return parser.parse_args()


def get_cifar_loaders(hparams: edict) -> (DataLoader, DataLoader, DataLoader, DataLoader):
	# Prepare data
	transform_fn = lambda img: np.array(img).transpose()  # Transpose img [3, 32, 32] to [32, 32, 3]
	train_set = CIFAR10(hparams.dataset, train=True, download=True, transform=transform_fn)
	val_set = CIFAR10(hparams.dataset, train=False, download=True, transform=transform_fn)

	# Create loaders
	sub_loaders_ratios = [hparams.supervised_ratio, 1.0 - hparams.supervised_ratio]

	cls_idx_all = get_classes_idx(train_set, hparams.nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	cls_idx_all = reduce_classes_idx(cls_idx_all, hparams.dataset_ratio)

	idx_train = split_classes_idx(cls_idx_all, sub_loaders_ratios)
	idx_train_s, idx_train_u = idx_train
	idx_val = list(range(int(len(val_set) * hparams.dataset_ratio)))

	process_fn = lambda batch, labels: (batch, one_hot(labels, hparams.nb_classes))
	loader_train_full = FnDataLoader(
		train_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_train_s + idx_train_u), num_workers=2,
		drop_last=True, fn=process_fn
	)
	loader_train_s = FnDataLoader(
		train_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_train_s), num_workers=2,
		drop_last=True, fn=process_fn
	)
	loader_train_u = NoLabelDataLoader(
		train_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_train_u), num_workers=2,
		drop_last=True
	)

	loader_val = FnDataLoader(
		val_set, batch_size=hparams.batch_size, sampler=SubsetRandomSampler(idx_val), num_workers=2, fn=process_fn
	)

	return loader_train_full, loader_train_s, loader_train_u, loader_val


def main():
	prog_start = time()

	args = create_args()

	hparams = edict()
	args_filtered = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in args.__dict__.items()}
	hparams.update(args_filtered)
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()

	reset_seed(hparams.seed)

	loader_train_full, loader_train_s, loader_train_u, loader_val = get_cifar_loaders(hparams)

	# Create model
	if hparams.model_name == "VGG11":
		model_factory = lambda: VGG("VGG11").cuda()
	elif hparams.model_name == "ResNet18":
		model_factory = lambda: ResNet18().cuda()
	else:
		raise RuntimeError("Unknown model %s" % hparams.model_name)

	if hparams.mode == "onehot":
		acti_fn = lambda batch, dim: batch.softmax(dim=dim)
	elif hparams.mode == "multihot":
		acti_fn = lambda batch, dim: batch.sigmoid()
	else:
		raise RuntimeError("Invalid argument")

	print("Model selected : %s (%d parameters, %d trainable parameters)." % (
		hparams.model_name, get_nb_parameters(model_factory()), get_nb_trainable_parameters(model_factory())))

	# Weak and strong augmentations used by FixMatch and ReMixMatch
	weak_augm_fn = to_batch_fn(RandomChoice([
		HorizontalFlip(0.5),
		VerticalFlip(0.5),
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	]))
	strong_augm_fn = to_batch_fn(Compose([
		RandomChoice([
			Transform(1.0, scale=(0.5, 1.5)),
			Transform(1.0, rotation=(-np.pi, np.pi)),
		]),
		RandomChoice([
			Gray(1.0),
			RandCrop(1.0),
			UniColor(1.0),
			Inversion(1.0),
		]),
	]))
	# Augmentation used by MixMatch
	augment_fn = to_batch_fn(RandomChoice([
		HorizontalFlip(0.5),
		VerticalFlip(0.5),
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
		Gray(0.5),
		RandCrop(0.5, rect_max_scale=(0.2, 0.2)),
		UniColor(0.5),
		Inversion(0.5),
	]))

	metrics_s = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u1 = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_r = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_val_lst = [
		CategoricalConfidenceAccuracy(hparams.confidence),
		MaxMetric(),
		FnMetric(cross_entropy),
	]
	metrics_val_names = ["acc", "max", "loss"]

	if "fm" in args.run:
		hparams_fm = default_fixmatch_hparams()
		hparams_fm.update(hparams)
		train_fixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, weak_augm_fn, strong_augm_fn,
			metrics_s, metrics_u, metrics_val_lst, metrics_val_names, hparams_fm
		)
	if "mm" in args.run:
		hparams_mm = default_mixmatch_hparams()
		hparams_mm.update(hparams)
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, augment_fn,
			metrics_s, metrics_u, metrics_val_lst, metrics_val_names, hparams_mm
		)
		hparams_mm.criterion_unsupervised = "crossentropy"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, augment_fn,
			metrics_s, metrics_u, metrics_val_lst, metrics_val_names, hparams_mm
		)
	if "rmm" in args.run:
		hparams_rmm = default_remixmatch_hparams()
		hparams_rmm.update(hparams)
		train_remixmatch(
			model_factory(), acti_fn, loader_train_s, loader_train_u, loader_val, weak_augm_fn, strong_augm_fn,
			metrics_s, metrics_u, metrics_u1, metrics_r, metrics_val_lst, metrics_val_names, hparams_rmm
		)

	if "sf" in args.run:
		hparams_sf = default_supervised_hparams()
		hparams_sf.update(hparams)
		train_supervised(
			model_factory(), acti_fn, loader_train_full, loader_val, metrics_s, metrics_val_lst, metrics_val_names,
			hparams_sf, suffix="full_100"
		)
	if "sp" in args.run:
		hparams_sp = default_supervised_hparams()
		hparams_sp.update(hparams)
		train_supervised(
			model_factory(), acti_fn, loader_train_s, loader_val, metrics_s, metrics_val_lst, metrics_val_names,
			hparams_sp, suffix="part_%d" % int(100 * hparams_sp.supervised_ratio)
		)

	exec_time = time() - prog_start
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (hparams.begin_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_nb_parameters(model: Module) -> int:
	return sum(p.numel() for p in model.parameters())


def get_nb_trainable_parameters(model: Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
	main()
