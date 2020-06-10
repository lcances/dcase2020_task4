import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import torch

from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomChoice, Compose

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip
from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor
from dcase2020_task4.util.other_metrics import CategoricalConfidenceAccuracy, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.utils_match import cross_entropy

from dcase2020_task4.resnet import ResNet18
from dcase2020_task4.fixmatch.train import train_fixmatch
from dcase2020_task4.mixmatch.train import train_mixmatch
from dcase2020_task4.remixmatch.train import train_remixmatch
from dcase2020_task4.supervised.train import train_supervised
from dcase2020_task4.fixmatch.hparams import default_fixmatch_hparams
from dcase2020_task4.mixmatch.hparams import default_mixmatch_hparams
from dcase2020_task4.remixmatch.hparams import default_remixmatch_hparams
from dcase2020_task4.supervised.hparams import default_supervised_hparams
from dcase2020_task4.vgg import VGG


def create_args() -> Namespace:
	# bool_fn = lambda x: str(x).lower() in ['true', '1', 'yes']  # TODO

	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf", "sp"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--dataset", type=str, default="../dataset/CIFAR10")
	parser.add_argument("--mode", type=str, default="onehot")
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--model_name", type=str, default="VGG11", choices=["VGG11", "ResNet18"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--dataset_ratio", type=float, default=1.0)
	parser.add_argument("--supervised_ratio", type=float, default=0.1)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.3)

	parser.add_argument("--num_workers_s", type=int, default=1)
	parser.add_argument("--num_workers_u", type=int, default=1)

	parser.add_argument("--lambda_u_max", type=float, default=10.0,
						help="MixMatch \"lambda_u\" hyperparameter.")
	parser.add_argument("--nb_augms", type=int, default=2,
						help="Nb of augmentations used in MixMatch.")
	parser.add_argument("--nb_augms_strong", type=int, default=2,
						help="Nb of strong augmentations used in ReMixMatch.")

	parser.add_argument("--threshold_mask", type=float, default=0.95,
						help="FixMatch threshold for compute mask.")
	parser.add_argument("--threshold_multihot", type=float, default=0.5,
						help="FixMatch threshold use to replace argmax() in multihot mode.")

	return parser.parse_args()


def main():
	prog_start = time()

	args = create_args()

	hparams = edict()
	args_filtered = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in args.__dict__.items()}
	hparams.update(args_filtered)
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()
	hparams.dataset_name = "CIFAR10"

	reset_seed(hparams.seed)

	# Create model
	if hparams.model_name == "VGG11":
		model_factory = lambda: VGG("VGG11").cuda()
	elif hparams.model_name == "ResNet18":
		model_factory = lambda: ResNet18().cuda()
	else:
		raise RuntimeError("Unknown model %s" % hparams.model_name)

	acti_fn = lambda batch, dim: batch.softmax(dim=dim)

	print("Model selected : %s (%d parameters, %d trainable parameters)." % (
		hparams.model_name, get_nb_parameters(model_factory()), get_nb_trainable_parameters(model_factory())))

	# Weak and strong augmentations used by FixMatch and ReMixMatch
	weak_augm_fn = RandomChoice([
		HorizontalFlip(0.5),
		VerticalFlip(0.5),
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	])
	strong_augm_fn = Compose([
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
	])
	# Augmentation used by MixMatch
	mm_ratio = 0.5
	augment_fn = RandomChoice([
		HorizontalFlip(mm_ratio),
		VerticalFlip(mm_ratio),
		Transform(mm_ratio, scale=(0.75, 1.25)),
		Transform(mm_ratio, rotation=(-np.pi, np.pi)),
		Gray(mm_ratio),
		RandCrop(mm_ratio, rect_max_scale=(0.2, 0.2)),
		UniColor(mm_ratio),
		Inversion(mm_ratio),
	])

	# Add preprocessing before each augmentation
	preprocess_fn = lambda img: np.array(img).transpose()  # Transpose img [3, 32, 32] to [32, 32, 3]

	metrics_s = {"acc_s": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_u = {"acc_u": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_u1 = {"acc_u1": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_r = {"acc_r": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_val = {
		"acc": CategoricalConfidenceAccuracy(hparams.confidence),
		"ce": FnMetric(cross_entropy),
		"eq": EqConfidenceMetric(hparams.confidence),
		"max": MaxMetric(),
	}

	# Prepare data
	dataset_train = CIFAR10(hparams.dataset, train=True, download=True, transform=preprocess_fn)
	dataset_val = CIFAR10(hparams.dataset, train=False, download=True, transform=preprocess_fn)

	dataset_train_weak = CIFAR10(
		hparams.dataset, train=True, download=True, transform=Compose([preprocess_fn, weak_augm_fn]))
	dataset_train_strong = CIFAR10(
		hparams.dataset, train=True, download=True, transform=Compose([preprocess_fn, strong_augm_fn]))
	dataset_train_augm = CIFAR10(
		hparams.dataset, train=True, download=True, transform=Compose([preprocess_fn, augment_fn]))

	# Compute sub-indexes for split CIFAR train dataset
	sub_loaders_ratios = [hparams.supervised_ratio, 1.0 - hparams.supervised_ratio]

	cls_idx_all = get_classes_idx(dataset_train, hparams.nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	cls_idx_all = reduce_classes_idx(cls_idx_all, hparams.dataset_ratio)
	idx_train = split_classes_idx(cls_idx_all, sub_loaders_ratios)

	idx_train_s, idx_train_u = idx_train
	idx_val = list(range(int(len(dataset_val) * hparams.dataset_ratio)))

	label_one_hot = lambda item: (item[0], one_hot(torch.as_tensor(item[1]), hparams.nb_classes).numpy())
	dataset_train = FnDataset(dataset_train, label_one_hot)
	dataset_val = FnDataset(dataset_val, label_one_hot)

	dataset_train_weak = FnDataset(dataset_train_weak, label_one_hot)
	dataset_train_strong = FnDataset(dataset_train_strong, label_one_hot)
	dataset_train_augm = FnDataset(dataset_train_augm, label_one_hot)

	dataset_val = Subset(dataset_val, idx_val)
	loader_val = DataLoader(dataset_val, batch_size=hparams.batch_size, shuffle=False, drop_last=True)

	args_loader_train_s = dict(
		batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers_u, drop_last=True)

	if "fm" in args.run:
		hparams_fm = default_fixmatch_hparams()
		hparams_fm.update(hparams)

		dataset_train_s_weak = Subset(dataset_train_weak, idx_train_s)
		dataset_train_u_weak = Subset(dataset_train_weak, idx_train_u)
		dataset_train_u_strong = Subset(dataset_train_strong, idx_train_u)

		dataset_train_u_weak = NoLabelDataset(dataset_train_u_weak)
		dataset_train_u_strong = NoLabelDataset(dataset_train_u_strong)

		dataset_train_u_weak_strong = MultipleDataset([dataset_train_u_weak, dataset_train_u_strong])

		loader_train_s_weak = DataLoader(dataset=dataset_train_s_weak, **args_loader_train_s)
		loader_train_u_weak_strong = DataLoader(dataset=dataset_train_u_weak_strong, **args_loader_train_u)

		train_fixmatch(
			model_factory(), acti_fn, loader_train_s_weak, loader_train_u_weak_strong, loader_val,
			metrics_s, metrics_u, metrics_val, hparams_fm
		)

	if "mm" in args.run:
		hparams_mm = default_mixmatch_hparams()
		hparams_mm.update(hparams)

		dataset_train_s_augm = Subset(dataset_train_augm, idx_train_s)
		dataset_train_u_augm = Subset(dataset_train_augm, idx_train_u)

		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)
		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * hparams_mm.nb_augms)

		loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
		loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

		# Train MixMatch with sqdiff for loss_u
		hparams_mm.criterion_name_u = "sqdiff"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s_augm, loader_train_u_augms, loader_val,
			metrics_s, metrics_u, metrics_val, hparams_mm
		)
		# Train MixMatch with crossentropy for loss_u
		hparams_mm.criterion_name_u = "crossentropy"
		train_mixmatch(
			model_factory(), acti_fn, loader_train_s_augm, loader_train_u_augms, loader_val,
			metrics_s, metrics_u, metrics_val, hparams_mm
		)

	if "rmm" in args.run:
		hparams_rmm = default_remixmatch_hparams()
		hparams_rmm.update(hparams)

		dataset_train_s_strong = Subset(dataset_train_strong, idx_train_s)
		dataset_train_u_weak = Subset(dataset_train_weak, idx_train_u)
		dataset_train_u_strong = Subset(dataset_train_strong, idx_train_u)

		dataset_train_u_weak = NoLabelDataset(dataset_train_u_weak)
		dataset_train_u_strong = NoLabelDataset(dataset_train_u_strong)

		dataset_train_u_strongs = MultipleDataset([dataset_train_u_strong] * hparams_rmm.nb_augms_strong)
		dataset_train_u_weak_strongs = MultipleDataset([dataset_train_u_weak, dataset_train_u_strongs])

		loader_train_s_strong = DataLoader(dataset_train_s_strong, **args_loader_train_s)
		loader_train_u_weak_strongs = DataLoader(dataset_train_u_weak_strongs, **args_loader_train_u)

		train_remixmatch(
			model_factory(), acti_fn, loader_train_s_strong, loader_train_u_weak_strongs, loader_val,
			metrics_s, metrics_u, metrics_u1, metrics_r, metrics_val, hparams_rmm
		)

	if "sf" in args.run:
		hparams_sf = default_supervised_hparams()
		hparams_sf.update(hparams)

		dataset_train_full = Subset(dataset_train, idx_train_s + idx_train_u)
		loader_train_full = DataLoader(dataset_train_full, **args_loader_train_s)

		train_supervised(
			model_factory(), acti_fn, loader_train_full, loader_val, metrics_s, metrics_val,
			hparams_sf, suffix="full_100"
		)

	if "sp" in args.run:
		hparams_sp = default_supervised_hparams()
		hparams_sp.update(hparams)

		dataset_train_s = Subset(dataset_train, idx_train_s)
		loader_train_s = DataLoader(dataset_train_s, **args_loader_train_s)

		train_supervised(
			model_factory(), acti_fn, loader_train_s, loader_val, metrics_s, metrics_val,
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
