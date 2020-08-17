"""
	Main script for testing MixMatch, ReMixMatch, FixMatch or supervised training on a mono-label dataset.
	Available datasets are CIFAR10 and UrbanSound8k.
	They do not have a supervised/unsupervised separation, so we need to split it manually.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


from argparse import ArgumentParser
from time import time
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomChoice, Compose

from augmentation_utils.signal_augmentations import TimeStretch, Occlusion, Noise, Noise2
from augmentation_utils.spec_augmentations import HorizontalFlip, RandomTimeDropout, RandomFreqDropout
from augmentation_utils.spec_augmentations import Noise as NoiseSpec

from dcase2020.util.utils import get_datetime, reset_seed

from dcase2020_task4.fixmatch.losses.tag.onehot import FixMatchLossOneHot
from dcase2020_task4.fixmatch.trainer import FixMatchTrainer
from dcase2020_task4.fixmatch.trainer_v11 import FixMatchTrainerV11

from dcase2020_task4.mixmatch.losses.tag.onehot import MixMatchLossOneHot
from dcase2020_task4.mixmatch.mixers.tag import MixMatchMixer
from dcase2020_task4.mixmatch.trainer import MixMatchTrainer
from dcase2020_task4.mixmatch.trainer_v3 import MixMatchTrainerV3

from dcase2020_task4.mixup.mixers.tag import MixUpMixerTag

from dcase2020_task4.remixmatch.losses.tag.onehot import ReMixMatchLossOneHot
from dcase2020_task4.remixmatch.mixers.tag import ReMixMatchMixer
from dcase2020_task4.remixmatch.self_label import SelfSupervisedFlips
from dcase2020_task4.remixmatch.trainer import ReMixMatchTrainer

from dcase2020_task4.supervised.trainer import SupervisedTrainer

from dcase2020_task4.util.avg_distributions import AvgDistributions
from dcase2020_task4.util.checkpoint import CheckPoint
from dcase2020_task4.util.datasets.dataset_idx import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, split_classes_idx
from dcase2020_task4.util.datasets.multiple_dataset import MultipleDataset
from dcase2020_task4.util.datasets.no_label_dataset import NoLabelDataset
from dcase2020_task4.util.datasets.onehot_dataset import OneHotDataset
from dcase2020_task4.util.datasets.random_choice_dataset import RandomChoiceDataset
from dcase2020_task4.util.datasets.smooth_dataset import SmoothOneHotDataset
from dcase2020_task4.util.datasets.to_tensor_dataset import ToTensorDataset
from dcase2020_task4.util.guessers.batch import *
from dcase2020_task4.util.other_img_augments import *
from dcase2020_task4.util.other_spec_augments import CutOutSpec
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot, MaxMetric, FnMetric, EqConfidenceMetric
from dcase2020_task4.util.ramp_up import RampUp
from dcase2020_task4.util.rand_augment import RandAugment
from dcase2020_task4.util.sharpen import Sharpen
from dcase2020_task4.util.types import str_to_bool, str_to_optional_str, str_to_union_str_int, str_to_optional_int
from dcase2020_task4.util.uniloss import ConstantEpochUniloss, WeightLinearUniloss, WeightLinearUnilossStepper
from dcase2020_task4.util.utils_match import cross_entropy
from dcase2020_task4.util.utils_standalone import *
from dcase2020_task4.util.zip_cycle import ZipCycle

from dcase2020_task4.learner import Learner
from dcase2020_task4.validator import ValidatorTag

from metric_utils.metrics import Metrics

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--run", type=str, default=None,
						choices=["fixmatch", "fm", "mixmatch", "mm", "remixmatch", "rmm", "supervised_full", "sf", "supervised_part", "sp"],
						help="Training method to run.")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--debug_mode", type=str_to_bool, default=False)
	parser.add_argument("--suffix", type=str, default="",
						help="Suffix to Tensorboard log dir.")

	parser.add_argument("--mode", type=str, default="onehot",
						choices=["onehot"])
	parser.add_argument("--dataset_path", type=str, default=osp.join("..", "dataset", "CIFAR10"), required=True)
	parser.add_argument("--dataset_name", type=str, default="CIFAR10",
						choices=["CIFAR10", "UBS8K"])
	parser.add_argument("--nb_classes", type=int, default=10)

	parser.add_argument("--logdir", type=str, default=osp.join("..", "..", "tensorboard"))
	parser.add_argument("--model", type=str, default="VGG11Rot",
						choices=["VGG11Rot", "ResNet18Rot", "WideResNetRot", "UBS8KBaselineRot", "CNN03Rot", "CNN03MishRot"])
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--confidence", type=float, default=0.5,
						help="Confidence threshold used in VALIDATION.")

	parser.add_argument("--batch_size_s", type=int, default=8,
						help="Batch size used for supervised loader.")
	parser.add_argument("--batch_size_u", type=int, default=8,
						help="Batch size used for unsupervised loader.")
	parser.add_argument("--num_workers_s", type=int, default=1,
						help="Number of workers created by supervised loader.")
	parser.add_argument("--num_workers_u", type=int, default=1,
						help="Number of workers created by unsupervised loader.")

	parser.add_argument("--optimizer", type=str, default="Adam",
						choices=["Adam", "SGD", "RAdam", "PlainRAdam", "AdamW"],
						help="Optimizer used.")
	parser.add_argument("--scheduler", type=str_to_optional_str, default="Cosine",
						choices=[None, "CosineLRScheduler", "Cosine"],
						help="FixMatch scheduler used. Use \"None\" for constant learning rate.")
	parser.add_argument("--lr", "--learning_rate", type=float, default=1e-3,
						help="Learning rate used.")
	parser.add_argument("--weight_decay", "--wd", type=float, default=0.0,
						help="Weight decay used.")

	parser.add_argument("--write_results", type=str_to_bool, default=True,
						help="Write results in a tensorboard SummaryWriter.")
	parser.add_argument("--args_file", type=str_to_optional_str, default=None,
						help="Filepath to args file. Values in this JSON will overwrite other options in terminal.")
	parser.add_argument("--checkpoint_path", type=str, default=osp.join("..", "models"),
						help="Directory path where checkpoint models will be saved.")
	parser.add_argument("--checkpoint_metric_name", type=str, default="acc",
						choices=["acc"],
						help="Metric used to compare and save best model during training.")

	parser.add_argument("--use_rampup", "--use_warmup", type=str_to_bool, default=False,
						help="Use RampUp or not for lambda_u and lambda_u1 hyperparameters.")
	parser.add_argument("--nb_rampup_steps", type=str_to_union_str_int, default="nb_epochs",
						help="Nb of steps when lambda_u and lambda_u1 is increase from 0 to their value."
							 "Use 0 for deactivate RampUp. Use \"nb_epochs\" for ramping up during all training.")
	parser.add_argument("--rampup_each_epoch", type=str_to_bool, default=True,
						help="If true, update RampUp each epoch, otherwise step each iteration.")

	parser.add_argument("--lambda_s", type=float, default=1.0,
						help="MixMatch, FixMatch and ReMixMatch \"lambda_s\" hyperparameter.")
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
	parser.add_argument("--history_size", type=int, default=128 * 64,
						help="Nb of predictions kept in AvgDistributions used in ReMixMatch.")

	parser.add_argument("--threshold_confidence", type=float, default=0.95,
						help="FixMatch threshold for compute confidence mask in loss.")
	parser.add_argument("--criterion_name_u", type=str, default="ce",
						choices=["sq_diff", "cross_entropy", "ce"],
						help="MixMatch unsupervised loss component.")

	parser.add_argument("--sharpen_temperature", "--temperature", type=float, default=0.5,
						help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening.")
	parser.add_argument("--mixup_alpha", "--alpha", type=float, default=0.75,
						help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp.")
	parser.add_argument("--mixup_distribution_name", type=str, default="beta",
						choices=["beta", "uniform", "constant"],
						help="MixUp distribution used in MixMatch and ReMixMatch.")
	parser.add_argument("--shuffle_s_with_u", type=str_to_bool, default=True,
						help="MixMatch shuffle supervised and unsupervised data.")

	parser.add_argument("--dataset_ratio", type=float, default=1.0,
						help="Ratio of the dataset used for training.")
	parser.add_argument("--supervised_ratio", type=float, default=0.1,
						help="Supervised ratio used for split dataset.")

	parser.add_argument("--cross_validation", type=str_to_bool, default=False,
						help="Use cross validation for UBS8K dataset.")
	parser.add_argument("--fold_val", type=int, default=10,
						help="Fold used for validation in UBS8K dataset. This parameter is unused if cross validation is True.")

	parser.add_argument("--ra_magnitude", type=str_to_optional_int, default=2,
						help="Magnitude used in RandAugment. Use \"None\" for generate a random "
							 "magnitude each time the augmentation is called.")
	parser.add_argument("--ra_nb_choices", type=int, default=1,
						help="Nb augmentations composed for RandAugment. ")

	parser.add_argument("--experimental", type=str_to_optional_str, default=None,
						choices=[None, "None", "V3", "V8", "V9", "V11", "V12"],
						help="Experimental mode activated.")

	parser.add_argument("--label_smoothing", type=float, default=0.0,
						help="Label smoothing value for supervised trainings. Use 0.0 for deactivate label smoothing.")
	parser.add_argument("--nb_classes_self_supervised", type=int, default=4,
						help="Nb classes in rotation loss (Self-Supervised part) of ReMixMatch.")

	parser.add_argument("--use_wlu", "--use_weight_linear_uniloss", type=str_to_bool, default=False,
						help="Activate Weight Linear Uniloss experimental mode.")
	parser.add_argument("--wlu_on_epoch", type=str_to_bool, default=True,
						help="Update WLU on iteration or on epoch.")
	parser.add_argument("--wlu_steps", type=int, default=10,
						help="Weight Linear Uniloss nb steps.")

	parser.add_argument("--dropout", type=float, default=0.5,
						help="Dropout used in model.")
	parser.add_argument("--wrn_depth", type=int, default=28,
						help="WideResNet widen factor.")
	parser.add_argument("--wrn_widen_factor", "--wrn_width", type=int, default=2,
						help="WideResNet widen factor.")

	parser.add_argument("--supervised_augment", type=str_to_optional_str, default=None,
						choices=[None, "weak", "strong"],
						help="Apply identity, weak or strong augment on supervised train dataset.")
	parser.add_argument("--standardize", type=str_to_bool, default=False,
						help="Normalize CIFAR10 data. Currently unused on UBS8K.")

	return parser.parse_args()


def main():
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	args = post_process_args(args)
	check_args(args)

	print("Start match_onehot. (suffix: \"%s\")" % args.suffix)
	print(" - start_date: %s" % start_date)

	print("Arguments :")
	for name, value in args.__dict__.items():
		print(" - %s: %s" % (name, str(value)))

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)

	metrics_s, metrics_u, metrics_u1, metrics_r, metrics_val = get_default_metrics(args)

	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)
	cross_validation_results = {}

	def run(fold_val_ubs8k: int):
		if args.dataset_name.lower() == "cifar10":
			augm_list_weak, augm_list_strong = get_cifar10_augms(args)
			dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong = \
				get_cifar10_datasets(args, augm_list_weak, augm_list_strong)
		elif args.dataset_name.lower() == "ubs8k":
			augm_list_weak, augm_list_strong = get_ubs8k_augms(args)
			dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong = \
				get_ubs8k_datasets(args, fold_val_ubs8k, augm_list_weak, augm_list_strong)
		else:
			raise RuntimeError("Unknown dataset %s" % args.dataset_name)

		sub_loaders_ratios = [args.supervised_ratio, 1.0 - args.supervised_ratio]

		# Compute sub-indexes for split train dataset
		cls_idx_all = get_classes_idx(dataset_train, args.nb_classes)
		cls_idx_all = shuffle_classes_idx(cls_idx_all)
		cls_idx_all = reduce_classes_idx(cls_idx_all, args.dataset_ratio)
		idx_train_s, idx_train_u = split_classes_idx(cls_idx_all, sub_loaders_ratios)

		idx_val = list(range(int(len(dataset_val) * args.dataset_ratio)))

		print("%s: %d train examples supervised, %d train examples unsupervised, %d validation examples" % (args.dataset_name, len(idx_train_s), len(idx_train_u), len(idx_val)))

		dataset_train = OneHotDataset(dataset_train, args.nb_classes)
		dataset_val = OneHotDataset(dataset_val, args.nb_classes)
		dataset_train_augm_weak = OneHotDataset(dataset_train_augm_weak, args.nb_classes)
		dataset_train_augm_strong = OneHotDataset(dataset_train_augm_strong, args.nb_classes)

		if args.label_smoothing > 0.0:
			dataset_train = SmoothOneHotDataset(dataset_train, args.nb_classes, args.label_smoothing)
			dataset_val = SmoothOneHotDataset(dataset_val, args.nb_classes, args.label_smoothing)
			dataset_train_augm_weak = SmoothOneHotDataset(dataset_train_augm_weak, args.nb_classes, args.label_smoothing)
			dataset_train_augm_strong = SmoothOneHotDataset(dataset_train_augm_strong, args.nb_classes, args.label_smoothing)

		dataset_val = Subset(dataset_val, idx_val)
		loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

		args_loader_train_s = dict(
			batch_size=args.batch_size_s, shuffle=True, num_workers=args.num_workers_s, drop_last=True)
		args_loader_train_u = dict(
			batch_size=args.batch_size_u, shuffle=True, num_workers=args.num_workers_u, drop_last=True)

		model = get_model_from_args(args)
		optim = get_optim_from_args(args, model)
		sched = get_sched_from_args(args, optim)
		print("Model selected : %s (%d parameters)." % (args.model, get_nb_parameters(model)))

		if args.write_results:
			writer = build_writer(args, start_date, "%d" % fold_val_ubs8k)
		else:
			writer = None

		if args.use_rampup:
			rampup_lambda_u = RampUp(args.nb_rampup_steps, args.lambda_u, obj=None, attr_name="lambda_u")
			rampup_lambda_u1 = RampUp(args.nb_rampup_steps, args.lambda_u1, obj=None, attr_name="lambda_u1")
			rampup_lambda_r = RampUp(args.nb_rampup_steps, args.lambda_r, obj=None, attr_name="lambda_r")
		else:
			rampup_lambda_u = None
			rampup_lambda_u1 = None
			rampup_lambda_r = None

		steppables_iteration = []
		steppables_epoch = []
		wlu_stepper = None

		if not args.rampup_each_epoch:
			steppables_iteration.append(rampup_lambda_u)

		if args.run in ["fm", "fixmatch"]:
			dataset_train_s_augm_weak = Subset(dataset_train_augm_weak, idx_train_s)

			dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)
			dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

			dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, idx_train_u)
			dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

			if args.experimental == "V11":
				dataset_train_u_augm_weak = MultipleDataset([dataset_train_u_augm_weak] * args.nb_augms)

			dataset_train_u_augms_weak_strong = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong])

			loader_train_s_augm_weak = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
			loader_train_u_augms_weak_strong = DataLoader(dataset=dataset_train_u_augms_weak_strong, **args_loader_train_u)
			loader = ZipCycle([loader_train_s_augm_weak, loader_train_u_augms_weak_strong])

			criterion = FixMatchLossOneHot.from_edict(args)
			if rampup_lambda_u is not None:
				rampup_lambda_u.set_obj(criterion)

			if args.use_wlu:
				nb_steps_wlu = args.nb_epochs * len(idx_train_u) * args.batch_size_u if not args.wlu_on_epoch else args.wlu_steps
				targets_wlu = [
					(criterion, "lambda_s", args.lambda_s, 1.0, 0.0),
					(criterion, "lambda_u", args.lambda_u, 0.0, 1.0),
				]
				wlu = WeightLinearUniloss(targets_wlu, nb_steps_wlu, False)
				wlu_stepper = WeightLinearUnilossStepper(args.nb_epochs, nb_steps_wlu, wlu)

				if not args.wlu_on_epoch:
					steppables_iteration.append(wlu_stepper)
				steppables_iteration.append(wlu)

			if args.experimental != "V11":
				if args.label_smoothing > 0.0:
					guesser = GuesserModelBinarizeSmooth(model, acti_fn, args.label_smoothing, args.nb_classes)
				else:
					guesser = GuesserModelBinarize(model, acti_fn)

				trainer = FixMatchTrainer(
					model, acti_fn, optim, loader, criterion, guesser, metrics_s, metrics_u,
					writer, steppables_iteration
				)
			else:
				guesser = GuesserMeanModelBinarize(model, acti_fn)
				trainer = FixMatchTrainerV11(
					model, acti_fn, optim, loader, criterion, guesser, metrics_s, metrics_u,
					writer, steppables_iteration
				)

		elif args.run in ["mm", "mixmatch"]:
			dataset_train_s_augm_weak = Subset(dataset_train_augm_weak, idx_train_s)
			dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)

			dataset_train_u_augm = NoLabelDataset(dataset_train_u_augm_weak)
			dataset_train_u_augms = MultipleDataset([dataset_train_u_augm] * args.nb_augms)

			if args.experimental == "V3":
				dataset_train_u = Subset(dataset_train, idx_train_u)
				dataset_train_u = NoLabelDataset(dataset_train_u)
				dataset_train_u_augms = MultipleDataset([dataset_train_u_augms, dataset_train_u])

			loader_train_s_augm = DataLoader(dataset=dataset_train_s_augm_weak, **args_loader_train_s)
			loader_train_u_augms = DataLoader(dataset=dataset_train_u_augms, **args_loader_train_u)
			loader = ZipCycle([loader_train_s_augm, loader_train_u_augms])

			if loader_train_s_augm.batch_size != loader_train_u_augms.batch_size:
				raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
					loader_train_s_augm.batch_size, loader_train_u_augms.batch_size))

			criterion = MixMatchLossOneHot.from_edict(args)
			if rampup_lambda_u is not None:
				rampup_lambda_u.set_obj(criterion)
			mixup_mixer = MixUpMixerTag.from_edict(args)
			mixer = MixMatchMixer(mixup_mixer, args.shuffle_s_with_u)

			sharpen_fn = Sharpen(args.sharpen_temperature)
			guesser = GuesserMeanModelSharpen(model, acti_fn, sharpen_fn)
			if not args.rampup_each_epoch:
				steppables_iteration.append(rampup_lambda_u)

			if args.experimental == "V8" or args.experimental == "V9":
				if args.use_rampup:
					raise RuntimeError("Experimental MMV8 (or MMV9) cannot be used with RampUp.")
				if args.nb_epochs < 10:
					raise RuntimeError("Cannot train with MMV8 (or MMV9) with less than %d epochs." % 10)

				begin_only_s = 0
				begin_uniform_s_u = int(args.nb_epochs * 0.1)
				begin_only_u = int(args.nb_epochs * 0.9)

				attributes = [(criterion, "lambda_s"), (criterion, "lambda_u")]

				if args.experimental == "V8":
					ratios_range = [
						([1.0, 0.0], begin_only_s, begin_uniform_s_u - 1),
						([0.5, 0.5], begin_uniform_s_u, begin_only_u - 1),
						([0.0, 1.0], begin_only_u, args.nb_epochs),
					]
				elif args.experimental == "V9":
					ratios_range = [
						([1.0, 0.0], begin_only_s, begin_uniform_s_u - 1),
						([0.5, 0.5], begin_uniform_s_u, args.nb_epochs),
					]
				else:
					raise RuntimeError("Invalid experimental mode %s" % args.experimental)

				constant_epoch_uniloss = ConstantEpochUniloss(attributes, ratios_range)
				steppables_epoch.append(constant_epoch_uniloss)

			if args.use_wlu:
				nb_steps_wlu = args.nb_epochs * len(idx_train_u) * args.batch_size_u if not args.wlu_on_epoch else args.wlu_steps
				targets_wlu = [
					(criterion, "lambda_s", args.lambda_s, 1.0, 0.0),
					(criterion, "lambda_u", args.lambda_u, 0.0, 1.0),
				]
				wlu = WeightLinearUniloss(targets_wlu, nb_steps_wlu, False)
				wlu_stepper = WeightLinearUnilossStepper(args.nb_epochs, nb_steps_wlu, wlu)

				if not args.wlu_on_epoch:
					steppables_iteration.append(wlu_stepper)
				steppables_iteration.append(wlu)

			if args.experimental != "V3":
				trainer = MixMatchTrainer(
					model, acti_fn, optim, loader, criterion, guesser, metrics_s, metrics_u,
					writer, mixer, steppables_iteration
				)
			else:
				trainer = MixMatchTrainerV3(
					model, acti_fn, optim, loader, criterion, guesser, metrics_s, metrics_u,
					writer, mixer, steppables_iteration
				)

		elif args.run in ["rmm", "remixmatch"]:
			dataset_train_s_augm_strong = Subset(dataset_train_augm_strong, idx_train_s)
			dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, idx_train_u)
			dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, idx_train_u)

			dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
			dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

			dataset_train_u_strongs = MultipleDataset([dataset_train_u_augm_strong] * args.nb_augms_strong)
			dataset_train_u_weak_strongs = MultipleDataset([dataset_train_u_augm_weak, dataset_train_u_strongs])

			loader_train_s_strong = DataLoader(dataset_train_s_augm_strong, **args_loader_train_s)
			loader_train_u_augms_weak_strongs = DataLoader(dataset_train_u_weak_strongs, **args_loader_train_u)
			loader = ZipCycle([loader_train_s_strong, loader_train_u_augms_weak_strongs])

			if loader_train_s_strong.batch_size != loader_train_u_augms_weak_strongs.batch_size:
				raise RuntimeError("Supervised and unsupervised batch size must be equal. (%d != %d)" % (
					loader_train_s_strong.batch_size, loader_train_u_augms_weak_strongs.batch_size)
				)

			criterion = ReMixMatchLossOneHot.from_edict(args)
			if rampup_lambda_u is not None:
				rampup_lambda_u.set_obj(criterion)
			if rampup_lambda_u1 is not None:
				rampup_lambda_u1.set_obj(criterion)
			if rampup_lambda_r is not None:
				rampup_lambda_r.set_obj(criterion)

			mixup_mixer = MixUpMixerTag.from_edict(args)
			mixer = ReMixMatchMixer(mixup_mixer, args.shuffle_s_with_u)

			sharpen_fn = Sharpen(args.sharpen_temperature)
			distributions = AvgDistributions.from_edict(args)
			guesser = GuesserModelAlignmentSharpen(model, acti_fn, distributions, sharpen_fn)

			acti_rot_fn = lambda batch, dim: batch.softmax(dim=dim).clamp(min=2e-30)

			# TODO : change conditions
			if args.dataset_name == "CIFAR10":
				ss_transform = SelfSupervisedFlips()
			elif args.dataset_name == "UBS8K":
				ss_transform = SelfSupervisedFlips()
			else:
				raise RuntimeError("Invalid argument \"mode = %s\". Use %s." % (
					args.dataset_name, " or ".join(("CIFAR10", "UBS8K"))
				))

			if ss_transform.get_nb_classes() != args.nb_classes_self_supervised:
				raise RuntimeError("Invalid self supervised transform.")

			if not args.rampup_each_epoch:
				steppables_iteration.append(rampup_lambda_u)
				steppables_iteration.append(rampup_lambda_u1)
				steppables_iteration.append(rampup_lambda_r)

			if args.use_wlu:
				nb_steps_wlu = args.nb_epochs * len(idx_train_u) * args.batch_size_u if not args.wlu_on_epoch else args.wlu_steps
				targets_wlu = [
					(criterion, "lambda_s", args.lambda_s, 1.0, 0.0),
					(criterion, "lambda_u", args.lambda_u, 0.0, 1.0 / 3.0),
					(criterion, "lambda_u1", args.lambda_u1, 0.0, 1.0 / 3.0),
					(criterion, "lambda_r", args.lambda_r, 0.0, 1.0 / 3.0),
				]
				wlu = WeightLinearUniloss(targets_wlu, nb_steps_wlu, False)
				wlu_stepper = WeightLinearUnilossStepper(args.nb_epochs, nb_steps_wlu, wlu)

				if not args.wlu_on_epoch:
					steppables_iteration.append(wlu_stepper)
				steppables_iteration.append(wlu)

			trainer = ReMixMatchTrainer(
				model, acti_fn, acti_rot_fn, optim, loader, criterion, guesser,
				metrics_s, metrics_u, metrics_u1, metrics_r,
				writer, mixer, distributions, ss_transform, steppables_iteration
			)

		elif args.run in ["sf", "supervised_full"]:
			if args.supervised_augment is None:
				dataset_train_full = dataset_train
			elif args.supervised_augment == "weak":
				dataset_train_full = dataset_train_augm_weak
			elif args.supervised_augment == "strong":
				dataset_train_full = dataset_train_augm_strong
			else:
				raise RuntimeError("Invalid supervised augment choice \"%s\"." % str(args.supervised_augment))

			dataset_train_full = Subset(dataset_train_full, idx_train_s + idx_train_u)
			loader_train_full = DataLoader(dataset_train_full, **args_loader_train_s)

			criterion = cross_entropy

			trainer = SupervisedTrainer(
				model, acti_fn, optim, loader_train_full, criterion, metrics_s, writer
			)

		elif args.run in ["sp", "supervised_part"]:
			if args.supervised_augment is None:
				dataset_train_part = dataset_train
			elif args.supervised_augment == "weak":
				dataset_train_part = dataset_train_augm_weak
			elif args.supervised_augment == "strong":
				dataset_train_part = dataset_train_augm_strong
			else:
				raise RuntimeError("Invalid supervised augment choice \"%s\"." % str(args.supervised_augment))

			dataset_train_part = Subset(dataset_train_part, idx_train_s)
			loader_train_part = DataLoader(dataset_train_part, **args_loader_train_s)

			criterion = cross_entropy

			trainer = SupervisedTrainer(
				model, acti_fn, optim, loader_train_part, criterion, metrics_s, writer
			)

		else:
			raise RuntimeError("Unknown run %s" % args.run)

		if args.write_results:
			filename_model = "%s_%s_%s.torch" % (args.model, args.train_name, args.suffix)
			filepath_model = osp.join(args.checkpoint_path, filename_model)
			checkpoint = CheckPoint(model, optim, name=filepath_model)
		else:
			checkpoint = None

		validator = ValidatorTag(
			model, acti_fn, loader_val, metrics_val, writer, checkpoint, args.checkpoint_metric_name
		)

		steppables_epoch.append(sched)
		if args.use_wlu and args.wlu_on_epoch:
			steppables_epoch.append(wlu_stepper)
		if args.use_rampup and args.rampup_each_epoch:
			steppables_epoch.append(rampup_lambda_u)
			steppables_epoch.append(rampup_lambda_u1)
			steppables_epoch.append(rampup_lambda_r)
		steppables_epoch = [steppable for steppable in steppables_epoch if steppable is not None]

		learner = Learner(args.train_name, trainer, validator, args.nb_epochs, steppables_epoch)
		learner.start()

		if writer is not None:
			augments_dict = {"augm_weak": augm_list_weak, "augm_strong": augm_list_strong}

			save_and_close_writer(writer, args, augments_dict)

			filepath_args = osp.join(writer.log_dir, "args.json")
			save_args(filepath_args, args)

			filepath_augms = osp.join(writer.log_dir, "augments.json")
			save_augms(filepath_augms, augments_dict)

		recorder = validator.get_metrics_recorder()
		recorder.print_min_max()

		maxs = recorder.get_maxs()
		cross_validation_results[fold_val_ubs8k] = maxs["acc"]

	if not args.cross_validation:
		run(args.fold_val)
	else:
		for fold_val_ubs8k_ in range(1, 11):
			run(fold_val_ubs8k_)

		content = [(" %d: %f" % (fold, value)) for fold, value in cross_validation_results.items()]
		mean_ = np.mean(list(cross_validation_results.values()))
		print("\n")
		print("Cross-validation results : \n", "\n".join(content))
		print("Cross-validation mean : ", mean_)

		if args.write_results:
			filepath = osp.join(args.logdir, "cross_val_results_%s.json" % start_date)
			content = {"results": cross_validation_results, "mean": mean_}
			with open(filepath, "w") as file:
				json.dump(content, file)

	exec_time = time() - start_time
	print("")
	print("Program started at \"%s\" and terminated at \"%s\"." % (start_date, get_datetime()))
	print("Total execution time: %.2fs" % exec_time)


def get_default_metrics(args: Namespace) -> List[Dict[str, Metrics]]:
	metrics_s = {
		"s_acc": CategoricalAccuracyOnehot(dim=1),
		"s_max": MaxMetric(dim=1),
	}
	metrics_u = {
		"u_acc": CategoricalAccuracyOnehot(dim=1),
		"u_max": MaxMetric(dim=1),
	}
	metrics_u1 = {
		"u1_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_r = {
		"r_acc": CategoricalAccuracyOnehot(dim=1),
	}
	metrics_val = {
		"acc": CategoricalAccuracyOnehot(dim=1),
		"ce": FnMetric(cross_entropy),
		"eq": EqConfidenceMetric(args.confidence, dim=1),
		"max": MaxMetric(dim=1),
	}
	return [metrics_s, metrics_u, metrics_u1, metrics_r, metrics_val]


def get_cifar10_augms(args: Namespace) -> (List[Callable], List[Callable]):
	ratio_augm_weak = 0.5
	augm_list_weak = [
		HorizontalFlip(ratio_augm_weak),
		# VerticalFlip(ratio_augm_weak),
		# Transform(ratio_augm_weak, scale=(0.75, 1.25)),
	]
	ratio_augm_strong = 1.0
	augm_list_strong = [
		CutOut(ratio_augm_strong),
		RandAugment(ratio=ratio_augm_strong, magnitude_m=args.ra_magnitude, nb_choices_n=args.ra_nb_choices),
	]

	return augm_list_weak, augm_list_strong


def get_cifar10_datasets(
	args: Namespace, augm_list_weak: List[Callable], augm_list_strong: List[Callable]) -> (Dataset, Dataset, Dataset, Dataset):
	# Add preprocessing before each augmentation
	pre_process_fn = lambda img: np.array(img)
	# Add postprocessing after each augmentation (shape : [32, 32, 3] -> [3, 32, 32])
	post_process_fn = lambda img: img.transpose()

	standardize_fn = Standardize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

	# Prepare TRAIN data
	transforms_train = [pre_process_fn, post_process_fn]
	if args.standardize:
		transforms_train.append(standardize_fn)
	dataset_train = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose(transforms_train))

	# Prepare VALIDATION data
	transforms_val = [pre_process_fn, post_process_fn]
	if args.standardize:
		transforms_val.append(standardize_fn)
	dataset_val = CIFAR10(
		args.dataset_path, train=False, download=True, transform=Compose(transforms_val))

	# Prepare WEAKLY AUGMENTED TRAIN data
	augm_weak_fn = RandomChoice(augm_list_weak)
	transforms_train_weak = [pre_process_fn, augm_weak_fn, post_process_fn]
	if args.standardize:
		transforms_train_weak.append(standardize_fn)
	dataset_train_augm_weak = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose(transforms_train_weak))

	# Prepare STRONGLY AUGMENTED TRAIN data
	augm_strong_fn = RandomChoice(augm_list_strong)
	transforms_train_strong = [pre_process_fn, augm_strong_fn, post_process_fn]
	if args.standardize:
		transforms_train_strong.append(standardize_fn)
	dataset_train_augm_strong = CIFAR10(
		args.dataset_path, train=True, download=True, transform=Compose(transforms_train_strong))

	return dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong


def get_ubs8k_augms(args: Namespace) -> (List[Callable], List[Callable]):
	ratio_augm_weak = 0.5
	augm_list_weak = [
		HorizontalFlip(ratio_augm_weak),
		Occlusion(ratio_augm_weak, max_size=1.0),
	]
	ratio_augm_strong = 1.0
	augm_list_strong = [
		TimeStretch(ratio_augm_strong),
		# PitchShiftRandom(ratio_augm_strong, steps=(-1, 1)),
		Noise(ratio_augm_strong, target_snr=15),
		CutOutSpec(ratio_augm_strong, rect_width_scale_range=(0.1, 0.25), rect_height_scale_range=(0.1, 0.25)),
		RandomTimeDropout(ratio_augm_strong, dropout=0.01),
		RandomFreqDropout(ratio_augm_strong, dropout=0.01),
		NoiseSpec(ratio_augm_strong, snr=5.0),
		Noise2(ratio_augm_strong, noise_factor=(0.005, 0.005)),
	]

	return augm_list_weak, augm_list_strong


def get_ubs8k_datasets(
	args: Namespace, fold_val: int, augm_list_weak: List[Callable], augm_list_strong: List[Callable]
) -> (Dataset, Dataset, Dataset, Dataset):
	metadata_root = osp.join(args.dataset_path, "metadata")
	audio_root = osp.join(args.dataset_path, "audio")

	folds_train = list(range(1, 11))
	folds_train.remove(fold_val)
	folds_train = tuple(folds_train)
	folds_val = (fold_val,)

	manager = UBS8KDatasetManager(metadata_root, audio_root)

	dataset_train = UBS8KDataset(manager, folds=folds_train, augments=(), cached=False, augment_choser=lambda x: x)
	dataset_train = ToTensorDataset(dataset_train)

	dataset_val = UBS8KDataset(manager, folds=folds_val, augments=(), cached=True, augment_choser=lambda x: x)
	dataset_val = ToTensorDataset(dataset_val)

	datasets = [UBS8KDataset(manager, folds=folds_train, augments=(augm_fn,), cached=False) for augm_fn in augm_list_weak]
	dataset_train_augm_weak = RandomChoiceDataset(datasets)
	dataset_train_augm_weak = ToTensorDataset(dataset_train_augm_weak)

	datasets = [UBS8KDataset(manager, folds=folds_train, augments=(augm_fn,), cached=False) for augm_fn in augm_list_strong]
	dataset_train_augm_strong = RandomChoiceDataset(datasets)
	dataset_train_augm_strong = ToTensorDataset(dataset_train_augm_strong)

	return dataset_train, dataset_val, dataset_train_augm_weak, dataset_train_augm_strong


if __name__ == "__main__":
	main()
