"""
	Main script for testing MixMatch, ReMixMatch, FixMatch or supervised training on a mono-label dataset.
	Available datasets are CIFAR10 and UrbanSound8k.
	They do not have a supervised/unsupervised separation, so we need to create it.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import numpy as np
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomChoice, Compose
from typing import Callable

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip, Noise, RandomTimeDropout, RandomFreqDropout

from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.fixmatch.losses.onehot import FixMatchLossOneHot
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer

from dcase2020_task4.mixmatch.losses.onehot import MixMatchLossOneHot
from dcase2020_task4.mixmatch.mixers.tag import MixMatchMixer
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer

from dcase2020_task4.other_models.resnet import ResNet18
from dcase2020_task4.other_models.UBS8KBaseline import UBS8KBaseline
from dcase2020_task4.other_models.vgg import VGG

from dcase2020_task4.remixmatch.losses.onehot import ReMixMatchLossOneHot
from dcase2020_task4.remixmatch.mixers.tag import ReMixMatchMixer
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer

from dcase2020_task4.supervised.trainer import SupervisedTrainer

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.util.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.util.FnDataset import FnDataset
from dcase2020_task4.util.MultipleDataset import MultipleDataset
from dcase2020_task4.util.NoLabelDataset import NoLabelDataset
from dcase2020_task4.util.other_augments import Gray, Inversion, RandCrop, UniColor
from dcase2020_task4.util.other_metrics import CategoricalConfidenceAccuracy, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.ramp_up import RampUp
from dcase2020_task4.util.sharpen import Sharpen
from dcase2020_task4.util.types import str_to_bool, str_to_optional_str
from dcase2020_task4.util.utils_match import cross_entropy, build_writer, filter_hparams, get_nb_parameters

from dcase2020_task4.learner import DefaultLearner
from dcase2020_task4.validator import DefaultValidator

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


def create_args() -> Namespace:
	available_methods = ", ".join([
		"fixmatch (fm)", "mixmatch (mm)", "remixmatch (rmm)", "supervised_full (sf)", "supervised_part (sp)"
	])

	parser = ArgumentParser()
	parser.add_argument("--run", type=str, nargs="*", default=["fixmatch"],
						help="Training method to run. Available method are : %s." % available_methods)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", type=str_to_bool, default=False)
	parser.add_argument("--begin_date", type=str, default=get_datetime(),
						help="Date used in SummaryWriter name.")

	parser.add_argument("--mode", type=str, default="onehot", choices=["onehot"])
	parser.add_argument("--dataset", type=str, default="../dataset/CIFAR10")
	parser.add_argument("--dataset_name", type=str, default="CIFAR10", choices=["CIFAR10", "UBS8K"])
	parser.add_argument("--nb_classes", type=int, default=10)

	parser.add_argument("--logdir", type=str, default="../../tensorboard")
	parser.add_argument("--model_name", type=str, default="VGG11", choices=["VGG11", "ResNet18", "UBS8KBaseline"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--confidence", type=float, default=0.3,
						help="Confidence threshold used in VALIDATION.")

	parser.add_argument("--batch_size_s", type=int, default=8,
						help="Batch size used for supervised loader.")
	parser.add_argument("--batch_size_u", type=int, default=8,
						help="Batch size used for unsupervised loader.")
	parser.add_argument("--num_workers_s", type=int, default=1,
						help="Number of workers created by supervised loader.")
	parser.add_argument("--num_workers_u", type=int, default=1,
						help="Number of workers created by unsupervised loader.")

	parser.add_argument("--optim_name", type=str, default="Adam", choices=["Adam", "SGD"],
						help="Optimizer used.")
	parser.add_argument("--scheduler", "--sched", type=str_to_optional_str, default="CosineLRScheduler",
						help="FixMatch scheduler used. Use \"None\" for constant learning rate.")
	parser.add_argument("--lr", type=float, default=1e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", type=float, default=0.0,
						help="Weight decay used.")

	parser.add_argument("--write_results", type=str_to_bool, default=True,
						help="Write results in a tensorboard SummaryWriter.")
	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")

	parser.add_argument("--dataset_ratio", type=float, default=1.0,
						help="Ratio of the dataset used for training.")
	parser.add_argument("--supervised_ratio", type=float, default=0.1,
						help="Supervised ratio used for split dataset.")
	parser.add_argument("--use_rampup", "--use_warmup", type=str_to_bool, default=False,
						help="Use RampUp or not for lambda_u FixMatch or MixMatch loss hyperparameter.")

	parser.add_argument("--lambda_u", type=float, default=1.0,
						help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter.")
	parser.add_argument("--lambda_u1", type=float, default=0.5,
						help="ReMixMatch \"lambda_u1\" hyperparameter.")
	parser.add_argument("--lambda_r", type=float, default=0.5,
						help="ReMixMatch \"lambda_r\" hyperparameter.")

	parser.add_argument("--nb_augms", type=int, default=2,
						help="Nb of augmentations used in MixMatch.")
	parser.add_argument("--nb_augms_strong", type=int, default=2,
						help="Nb of strong augmentations used in ReMixMatch.")
	parser.add_argument("--history_size", type=int, default=128,
						help="Nb of prediction kept in AvgDistributions used in ReMixMatch.")

	parser.add_argument("--threshold_confidence", type=float, default=0.95,
						help="FixMatch threshold for compute confidence mask in loss.")
	parser.add_argument("--criterion_name_u", type=str, default="cross_entropy", choices=["sq_diff", "cross_entropy"],
						help="MixMatch unsupervised loss component.")

	parser.add_argument("--sharpen_temperature", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening.")
	parser.add_argument("--mixup_alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")

	return parser.parse_args()


def check_args(args: Namespace):
	if not osp.isdir(args.dataset):
		raise RuntimeError("Invalid dirpath %s" % args.dataset)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath %s" % args.logdir)

	if args.dataset_name == "CIFAR10":
		if args.model_name not in ["VGG11", "ResNet18"]:
			raise RuntimeError("Invalid model %s for dataset %s" % (args.model_name, args.dataset_name))
	elif args.dataset_name == "UBS8K":
		if args.model_name not in ["UBS8KBaseline"]:
			raise RuntimeError("Invalid model %s for dataset %s" % (args.model_name, args.dataset_name))


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	print("Start match_onehot. (%s)" % args.suffix)
	print("- run:", " ".join(args.run))
	print("- confidence:", args.confidence)

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	metrics_s = {
		"s_acc": CategoricalConfidenceAccuracy(args.confidence),
	}
	metrics_u = {
		"u_acc": CategoricalConfidenceAccuracy(args.confidence),
	}
	metrics_u1 = {
		"u1_acc": CategoricalConfidenceAccuracy(args.confidence),
	}
	metrics_r = {
		"r_acc": CategoricalConfidenceAccuracy(args.confidence),
	}
	metrics_val = {
		"acc": CategoricalConfidenceAccuracy(args.confidence),
		"ce": FnMetric(cross_entropy),
		"eq": EqConfidenceMetric(args.confidence),
		"max": MaxMetric(),
	}

	def model_factory() -> Module:
		if args.model_name.lower() == "vgg11":
			return VGG("VGG11").cuda()
		elif args.model_name.lower() == "resnet18":
			return ResNet18().cuda()
		elif args.model_name.lower() == "ubs8kbaseline":
			return UBS8KBaseline().cuda()
		else:
			raise RuntimeError("Unknown model %s" % args.model_name)

	def optim_factory(model_: Module) -> Optimizer:
		if args.optim_name.lower() == "adam":
			return Adam(model_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		elif args.optim_name.lower() == "sgd":
			return SGD(model_.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		else:
			raise RuntimeError("Unknown optimizer %s" % str(args.optim_name))

	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	if args.dataset_name.lower() == "cifar10":
		dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm = \
			get_cifar10_datasets(args)
	elif args.dataset_name.lower() == "ubs8k":
		dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm = \
			get_ubs8k_datasets(args)
	else:
		raise RuntimeError("Unknown dataset %s" % args.dataset_name)

	sub_loaders_ratios = [args.supervised_ratio, 1.0 - args.supervised_ratio]

	# Compute sub-indexes for split train dataset
	cls_idx_all = get_classes_idx(dataset_train, args.nb_classes)
	cls_idx_all = shuffle_classes_idx(cls_idx_all)
	cls_idx_all = reduce_classes_idx(cls_idx_all, args.dataset_ratio)
	idx_train_s, idx_train_u = split_classes_idx(cls_idx_all, sub_loaders_ratios)

	idx_val = list(range(int(len(dataset_val) * args.dataset_ratio)))

	label_one_hot = lambda item: (item[0], one_hot(torch.as_tensor(item[1]), args.nb_classes).numpy())
	dataset_train = FnDataset(dataset_train, label_one_hot)
	dataset_val = FnDataset(dataset_val, label_one_hot)

	dataset_train_augm_weak = FnDataset(dataset_train_augm_weak, label_one_hot)
	dataset_train_augm_strong = FnDataset(dataset_train_augm_strong, label_one_hot)
	dataset_train_augm = FnDataset(dataset_train_augm, label_one_hot)

	dataset_val = Subset(dataset_val, idx_val)
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

	args_loader_train_s = dict(
		batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)
	args_loader_train_u = dict(
		batch_size=args.batch_size_u, shuffle=True, num_workers=args.num_workers_u, drop_last=True)

	if "fm" in args.run or "fixmatch" in args.run:
		args.train_name = "FixMatch"
		dataset_train_s_augm_weak = Subset(dataset_train_augm_weak, idx_train_s)
		dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)
		dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, idx_train_u)

		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_augms_weak_strong = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong])

		loader_train_s_augm_weak = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
		loader_train_u_augms_weak_strong = DataLoader(dataset=dataset_train_u_augms_weak_strong, **args_loader_train_u)

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		if args.scheduler == "CosineLRScheduler":
			scheduler = CosineLRScheduler(optim, nb_epochs=args.nb_epochs, lr0=args.lr)
		else:
			scheduler = None

		criterion = FixMatchLossOneHot.from_edict(args)

		if args.write_results:
			writer = build_writer(args, suffix="%d_%d_%s_%.2f_%.2f_%s" % (
				args.batch_size_s, args.batch_size_u, str(args.scheduler), args.threshold_confidence, args.lambda_u, args.suffix))
		else:
			writer = None

		if args.use_rampup:
			nb_rampup_steps = args.nb_epochs * len(loader_train_u_augms_weak_strong)
			rampup_lambda_u = RampUp(args.lambda_u, nb_rampup_steps)
		else:
			rampup_lambda_u = None

		trainer = FixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm_weak, loader_train_u_augms_weak_strong, metrics_s, metrics_u,
			criterion, writer, args.mode, rampup_lambda_u
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs, scheduler)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args.__dict__, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.close()

	if "mm" in args.run or "mixmatch" in args.run:
		args.train_name = "MixMatch"
		dataset_train_s_augm = Subset(dataset_train_augm, idx_train_s)
		dataset_train_u_augm = Subset(dataset_train_augm, idx_train_u)

		dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm)
		dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * args.nb_augms)

		loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm, **args_loader_train_s)
		loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)

		if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = MixMatchLossOneHot.from_edict(args)
		mixer = MixMatchMixer(args.mixup_alpha)
		sharpen_fn = Sharpen(args.sharpen_temperature)

		nb_rampup_steps = args.nb_epochs * len(loader_train_u_augms)
		rampup_lambda_u = RampUp(args.lambda_u, nb_rampup_steps)

		if args.write_results:
			writer = build_writer(args, suffix="%d_%d_%s_%.2f_%s" % (
				args.batch_size_s, args.batch_size_u, args.criterion_name_u, args.lambda_u, args.suffix))
		else:
			writer = None

		trainer = MixMatchTrainer(
			model, acti_fn, optim, loader_train_s_augm, loader_train_u_augms, metrics_s, metrics_u,
			criterion, writer, mixer, rampup_lambda_u, sharpen_fn
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args.__dict__, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.close()

	if "rmm" in args.run or "remixmatch" in args.run:
		args.train_name = "ReMixMatch"
		dataset_train_s_augm_strong = Subset(dataset_train_augm_strong, idx_train_s)
		dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)
		dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, idx_train_u)

		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		dataset_train_u_strongs = MultipleDataset([dataset_train_u_augm_strong] * args.nb_augms_strong)
		dataset_train_u_weak_strongs = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_strongs])

		loader_train_s_strong = DataLoader(dataset_train_s_augm_strong, **args_loader_train_s)
		loader_train_u_augms_weak_strongs = DataLoader(dataset_train_u_weak_strongs, **args_loader_train_u)

		if loader_train_s_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
			raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
				loader_train_s_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size))

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		rot_angles = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

		criterion = ReMixMatchLossOneHot.from_edict(args)
		mixer = ReMixMatchMixer(args.mixup_alpha)
		sharpen_fn = Sharpen(args.sharpen_temperature)

		distributions = AvgDistributions.from_edict(args)
		acti_rot_fn = lambda batch, dim: batch.softmax(dim=dim).clamp(min=2e-30)

		if args.write_results:
			writer = build_writer(args, suffix="%d_%d_%.2f_%.2f_%.2f_%s" % (
				args.batch_size_s, args.batch_size_u, args.lambda_u, args.lambda_u1, args.lambda_r, args.suffix))
		else:
			writer = None

		trainer = ReMixMatchTrainer(
			model, acti_fn, acti_rot_fn, optim, loader_train_s_strong, loader_train_u_augms_weak_strongs, metrics_s, metrics_u,
			metrics_u1, metrics_r, criterion, writer, mixer, distributions, rot_angles, sharpen_fn,
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args.__dict__, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.close()

	if "sf" in args.run or "supervised_full" in args.run:
		args.train_name = "Supervised"
		dataset_train_full = Subset(dataset_train, idx_train_s + idx_train_u)
		loader_train_full = DataLoader(dataset_train_full, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = cross_entropy

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%s" % ("full_100", args.batch_size_s, args.batch_size_u, args.suffix))
		else:
			writer = None

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train_full, metrics_s, criterion, writer
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args.__dict__, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.close()

	if "sp" in args.run or "supervised_part" in args.run:
		args.train_name = "Supervised"
		dataset_train_part = Subset(dataset_train, idx_train_s)
		loader_train_part = DataLoader(dataset_train_part, **args_loader_train_s)

		model = model_factory()
		optim = optim_factory(model)
		print("Model selected : %s (%d parameters)." % (args.model_name, get_nb_parameters(model)))

		criterion = cross_entropy

		if args.write_results:
			writer = build_writer(args, suffix="%s_%d_%d_%d_%s" % (
				"part", int(100 * args.supervised_ratio), args.batch_size_s, args.batch_size_u, args.suffix))
		else:
			writer = None

		trainer = SupervisedTrainer(
			model, acti_fn, optim, loader_train_part, metrics_s, criterion, writer
		)
		validator = DefaultValidator(
			model, acti_fn, loader_val, metrics_val, writer
		)
		learner = DefaultLearner(args.train_name, trainer, validator, args.nb_epochs)
		learner.start()

		if writer is not None:
			with open(osp.join(writer.log_dir, "args.json"), "w") as file:
				json.dump(args.__dict__, file, indent="\t")
			writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
			writer.close()

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_cifar10_augms() -> (Callable, Callable, Callable):
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	augm_weak_fn = RandomChoice([
		HorizontalFlip(0.5),
		VerticalFlip(0.5),
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	])
	augm_strong_fn = Compose([
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
	augm_fn = RandomChoice([
		HorizontalFlip(mm_ratio),
		VerticalFlip(mm_ratio),
		Transform(mm_ratio, scale=(0.75, 1.25)),
		Transform(mm_ratio, rotation=(-np.pi, np.pi)),
		Gray(mm_ratio),
		RandCrop(mm_ratio, rect_max_scale=(0.2, 0.2)),
		UniColor(mm_ratio),
		Inversion(mm_ratio),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


def get_cifar10_datasets(args: Namespace) -> (Dataset, Dataset, Dataset, Dataset, Dataset):
	augm_weak_fn, augm_strong_fn, augm_fn = get_cifar10_augms()

	# Add preprocessing before each augmentation
	preprocess_fn = lambda img: np.array(img).transpose()  # Transpose img [3, 32, 32] to [32, 32, 3]

	# Prepare data
	dataset_train = CIFAR10(args.dataset, train=True, download=True, transform=preprocess_fn)
	dataset_val = CIFAR10(args.dataset, train=False, download=True, transform=preprocess_fn)

	dataset_train_augm_weak = CIFAR10(
		args.dataset, train=True, download=True, transform=Compose([preprocess_fn, augm_weak_fn]))
	dataset_train_augm_strong = CIFAR10(
		args.dataset, train=True, download=True, transform=Compose([preprocess_fn, augm_strong_fn]))
	dataset_train_augm = CIFAR10(
		args.dataset, train=True, download=True, transform=Compose([preprocess_fn, augm_fn]))

	return dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm


def get_ubs8k_augms() -> (Callable, Callable, Callable):
	# Weak and strong augmentations used by FixMatch and ReMixMatch
	ratio = 1.0
	# TimeStretch(ratio),
	augm_weak_fn = RandomChoice([
		RandomChoice([
			PitchShiftRandom(ratio, steps=(-1, 1)),
			Noise(ratio=ratio, snr=5.0),
			Noise2(ratio, noise_factor=(5.0, 5.0)),
		]),
	])
	augm_strong_fn = Compose([
		RandomChoice([
			PitchShiftRandom(ratio),
			Noise(ratio=ratio, snr=15.0),
			Noise2(ratio, noise_factor=(10.0, 10.0)),
		]),
		RandomChoice([
			Occlusion(ratio, max_size=1.0),
			RandomFreqDropout(ratio, dropout=0.5),
			RandomTimeDropout(ratio, dropout=0.5),
		]),
	])
	augm_fn = RandomChoice([
		TimeStretch(ratio),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		Noise(ratio=ratio, snr=15.0),
		Noise2(ratio, noise_factor=(10.0, 10.0)),
		RandomFreqDropout(ratio, dropout=0.5),
		RandomTimeDropout(ratio, dropout=0.5),
	])

	return augm_weak_fn, augm_strong_fn, augm_fn


def get_ubs8k_datasets(args: Namespace) -> (Dataset, Dataset, Dataset, Dataset, Dataset):
	augm_weak_fn, augm_strong_fn, augm_fn = get_ubs8k_augms()
	metadata_root = osp.join(args.dataset, "metadata")
	audio_root = osp.join(args.dataset, "audio")

	folds_train = (1, 2, 3, 4, 5, 6, 7, 8, 9)
	folds_val = (10,)

	manager = UBS8KDatasetManager(metadata_root, audio_root)

	# Shapes : (64, 173), (1)
	dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(), cached=False)
	dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True)

	dataset_train_augm_weak = UBS8KDataset(manager, folds=folds_train, augments=(augm_weak_fn,), cached=False)
	dataset_train_augm_strong = UBS8KDataset(manager, folds=folds_train, augments=(augm_strong_fn,), cached=False)
	dataset_train_augm = UBS8KDataset(manager, folds=folds_train, augments=(augm_fn,), cached=False)

	return dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong, dataset_train_augm


if __name__ == "__main__":
	main()
