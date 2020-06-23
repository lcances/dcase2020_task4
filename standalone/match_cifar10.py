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
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomChoice, Compose

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip
from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.fixmatch.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.fixmatch.losses.onehot import FixMatchLossOneHot
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer

from dcase2020_task4.mixmatch.losses.onehot import MixMatchLossOneHot
from dcase2020_task4.mixmatch.mixers.tag_loc import MixMatchMixer
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer

from dcase2020_task4.remixmatch.losses.onehot import ReMixMatchLossOneHot
from dcase2020_task4.remixmatch.mixer import ReMixMatchMixer
from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer

from dcase2020_task4.supervised.trainer import SupervisedTrainer

from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor
from dcase2020_task4.util.other_metrics import CategoricalConfidenceAccuracy, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.rampup import RampUp
from dcase2020_task4.util.utils_match import cross_entropy, build_writer

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.resnet import ResNet18
from dcase2020_task4.validator import DefaultValidator
from dcase2020_task4.vgg import VGG


def create_args() -> Namespace:
	# bool_fn = lambda x: str(x).lower() in ['true', '1', 'yes']  # TODO

	parser = ArgumentParser()
	# TODO : help for acronyms
	parser.add_argument("--run", type=str, nargs="*", default=["fm", "mm", "rmm", "sf", "sp"])
	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--dataset", type=str, default="../dataset/CIFAR10")
	parser.add_argument("--mode", type=str, default="onehot")
	parser.add_argument("--dataset_name", type=str, default="CIFAR10")
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--model_name", type=str, default="VGG11", choices=["VGG11", "ResNet18"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--dataset_ratio", type=float, default=1.0)
	parser.add_argument("--supervised_ratio", type=float, default=0.1)
	parser.add_argument("--batch_size_s", type=int, default=8)
	parser.add_argument("--batch_size_u", type=int, default=8)
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--confidence", type=float, default=0.3)

	parser.add_argument("--num_workers_s", type=int, default=1)
	parser.add_argument("--num_workers_u", type=int, default=1)

	parser.add_argument("--lambda_u", type=float, default=10.0,
						help="MixMatch \"lambda_u\" hyperparameter.")
	parser.add_argument("--nb_augms", type=int, default=2,
						help="Nb of augmentations used in MixMatch.")
	parser.add_argument("--nb_augms_strong", type=int, default=2,
						help="Nb of strong augmentations used in ReMixMatch.")

	parser.add_argument("--threshold_mask", type=float, default=0.95,
						help="FixMatch threshold for compute mask.")
	parser.add_argument("--threshold_multihot", type=float, default=0.5,
						help="FixMatch threshold use to replace argmax() in multihot mode.")

	parser.add_argument("--lr", type=float, default=1e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", type=float, default=0.0,
						help="Weight decay used.")
	parser.add_argument("--optim_name", type=str, default="Adam", choices=["Adam"],
						help="Optimizer used.")
	parser.add_argument("--criterion_name_u", type=str, default="cross_entropy", choices=["sq_diff", "cross_entropy"],
						help="MixMatch unsupervised loss component.")

	parser.add_argument("--sharpen_temp", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening.")
	parser.add_argument("--mixup_alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")

	return parser.parse_args()


def main():
	prog_start = time()

	args = create_args()
	print("Start match_cifar10.")
	print("- run:", " ".join(args.run))

	hparams = edict()
	hparams.update({
		k: (str(v) if v is None else (" ".join(v) if isinstance(v, list) else v))
		for k, v in args.__dict__.items()
	})
	# Note : some hyperparameters are overwritten when calling the training function, change this in the future
	hparams.begin_date = get_datetime()

	reset_seed(hparams.seed)

	# Create model
	if hparams.model_name == "VGG11":
		model_factory = lambda: VGG("VGG11").cuda()
	elif hparams.model_name == "ResNet18":
		model_factory = lambda: ResNet18().cuda()
	else:
		raise RuntimeError("Unknown model %s" % hparams.model_name)

	acti_fn = lambda batch, dim: batch.softmax(dim=dim)

	def optim_factory(model: Module) -> Optimizer:
		if hparams.optim_name.lower() == "adam":
			return Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
		else:
			raise RuntimeError("Unknown optimizer %s" % str(hparams.optim_name))

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

	metrics_s = {"s_acc": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_u = {"u_acc": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_u1 = {"u1_acc": CategoricalConfidenceAccuracy(hparams.confidence)}
	metrics_r = {"r_acc": CategoricalConfidenceAccuracy(hparams.confidence)}
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
		batch_size=hparams.batch_size_s, shuffle=True, num_workers=hparams.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=hparams.batch_size_u, shuffle=True, num_workers=hparams.num_workers_u, drop_last=True)

	if "fm" in args.run or "fixmatch" in args.run:
		dataset_train_s_augm_weak = Subset(dataset_train_weak, idx_train_s)
		dataset_train_u_weak = Subset(dataset_train_weak, idx_train_u)
		dataset_train_u_strong = Subset(dataset_train_strong, idx_train_u)

		dataset_train_u_weak = NoLabelDataset(dataset_train_u_weak)
		dataset_train_u_strong = NoLabelDataset(dataset_train_u_strong)

		dataset_train_u_weak_strong = MultipleDataset([dataset_train_u_weak, dataset_train_u_strong])

		loader_train_s_augm_weak = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
		loader_train_u_augms_weak_strong = DataLoader(dataset=dataset_train_u_weak_strong, **args_loader_train_u)

		model = model_factory()
		optim = optim_factory(model)

		if hparams.scheduler == "CosineLRScheduler":
			scheduler = CosineLRScheduler(optim, nb_epochs=hparams.nb_epochs, lr0=hparams.lr)
		else:
			scheduler = None

		hparams.train_name = "FixMatch"
		writer = build_writer(hparams, suffix="%s_%s" % (str(hparams.scheduler), hparams.suffix))

		criterion = FixMatchLossOneHot.from_edict(hparams)

		trainer = FixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
			criterion, writer, hparams.mode, hparams.threshold_multihot
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs, scheduler)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "mm" in args.run or "mixmatch" in args.run:
		dataset_train_s_augm = Subset(dataset_train_augm, idx_train_s)
		dataset_train_u_augm = Subset(dataset_train_augm, idx_train_u)

		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)
		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * hparams.nb_augms)

		loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
		loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

		if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

		model = model_factory()
		optim = optim_factory(model)

		hparams.train_name = "MixMatch"
		writer = build_writer(hparams, suffix=hparams.criterion_name_u)

		nb_rampup_steps = hparams.nb_epochs * len(loader_train_u_augms)

		criterion = MixMatchLossOneHot.from_edict(hparams)
		mixer = MixMatchMixer(model, acti_fn, hparams.nb_augms, hparams.sharpen_temp, hparams.mixup_alpha,
							  hparams.sharpen_threshold_multihot)
		lambda_u_rampup = RampUp(hparams.lambda_u, nb_rampup_steps)

		trainer = MixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
			criterion, writer, mixer, lambda_u_rampup
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "rmm" in args.run or "remixmatch" in args.run:
		dataset_train_s_augm_strong = Subset(dataset_train_strong, idx_train_s)
		dataset_train_u_weak = Subset(dataset_train_weak, idx_train_u)
		dataset_train_u_strong = Subset(dataset_train_strong, idx_train_u)

		dataset_train_u_weak = NoLabelDataset(dataset_train_u_weak)
		dataset_train_u_strong = NoLabelDataset(dataset_train_u_strong)

		dataset_train_u_strongs = MultipleDataset([dataset_train_u_strong] * hparams.nb_augms_strong)
		dataset_train_u_weak_strongs = MultipleDataset([dataset_train_u_weak, dataset_train_u_strongs])

		loader_train_s_strong = DataLoader(dataset_train_s_augm_strong, **args_loader_train_s)
		loader_train_u_augms_weak_strongs = DataLoader(dataset_train_u_weak_strongs, **args_loader_train_u)

		if loader_train_s_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size))

		model = model_factory()
		optim = optim_factory(model)

		rot_angles = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
		hparams.train_name = "ReMixMatch"
		writer = build_writer(hparams)

		criterion = ReMixMatchLossOneHot.from_edict(hparams)
		distributions = AvgDistributions(
			history_size=hparams.history_size,
			nb_classes=hparams.nb_classes,
			mode=hparams.mode,
			names=["labeled", "unlabeled"],
		)
		mixer = ReMixMatchMixer(
			model,
			acti_fn,
			distributions,
			hparams.nb_augms_strong,
			hparams.sharpen_temp,
			hparams.mixup_alpha,
			hparams.mode
		)
		trainer = ReMixMatchTrainer(
			model, acti_fn, optim, loader_train_s_strong, loader_train_u_augms_weak_strongs, metrics_s, metrics_u,
			metrics_u1, metrics_r, criterion, writer, mixer, distributions, rot_angles
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "sf" in args.run or "supervised_full" in args.run:
		dataset_train_full = Subset(dataset_train, idx_train_s + idx_train_u)
		loader_train_full = DataLoader(dataset_train_full, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)

		hparams.train_name = "Supervised"
		writer = build_writer(hparams, suffix="full_100")

		criterion = cross_entropy

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train_full, metrics_s, criterion, writer
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

	if "sp" in args.run or "supervised_part" in args.run:
		dataset_train_part = Subset(dataset_train, idx_train_s)
		loader_train_part = DataLoader(dataset_train_part, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)

		hparams.train_name = "Supervised"
		writer = build_writer(hparams, suffix="part_%d" % int(100 * hparams.supervised_ratio))

		criterion = cross_entropy

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train_part, metrics_s, criterion, writer
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(hparams.train_name, trainer, validator, hparams.nb_epochs)
		learner.start()

		writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
		writer.close()

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
