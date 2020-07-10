"""
	Methods used in several standalone files for training MixMatch, ReMixMatch or FixMatch.
"""

import json
import os.path as osp

from argparse import Namespace
from torch.nn import Module
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from dcase2020_task4.dcase2019.models import dcase2019_model
from dcase2020_task4.other_models.weak_baseline_rot import WeakBaselineRot, WeakStrongBaselineRot
from dcase2020_task4.other_models.vgg import VGG
from dcase2020_task4.other_models.resnet import ResNet18
from dcase2020_task4.other_models.UBS8KBaseline import UBS8KBaseline
from dcase2020_task4.util.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.validator_abc import ValidatorABC


def check_args(args: Namespace):
	if not osp.isdir(args.dataset_path):
		raise RuntimeError("Invalid dirpath %s" % args.dataset_path)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath %s" % args.logdir)
		if not osp.isdir(args.checkpoint_path):
			raise RuntimeError("Invalid dirpath %s" % args.checkpoint_path)

	if args.dataset_name == "CIFAR10":
		if args.model_name not in ["VGG11", "ResNet18"]:
			raise RuntimeError("Invalid model %s for dataset %s" % (args.model_name, args.dataset_name))
		if args.cross_validation:
			raise RuntimeError("Cross-validation on %s dataset is not supported." % args.dataset_name)

	elif args.dataset_name == "UBS8K":
		if args.model_name not in ["UBS8KBaseline"]:
			raise RuntimeError("Invalid model %s for dataset %s" % (args.model_name, args.dataset_name))
		if not(1 <= args.fold_val <= 10):
			raise RuntimeError("Invalid fold %d (must be in [%d,%d])" % (args.fold_val, 1, 10))


def post_process_args(args: Namespace) -> Namespace:
	if args.args_file is not None:
		if not osp.isfile(args.args_file):
			raise RuntimeError("Unknown file \"%s\"." % args.args_file)

		with open(args.args_file, "r") as file:
			args_dict = json.load(file)
			differences = set(args_dict.keys()).difference(args.__dict__.keys())
			if len(differences) > 0:
				raise RuntimeError("Found unknown(s) key(s) in JSON file : %s" % ", ".join(differences))
			args.__dict__.update(args_dict)

	if args.nb_rampup_epochs == "nb_epochs":
		args.nb_rampup_epochs = args.nb_epochs
	args.train_name = run_to_train_name(args.run)
	return args


def model_factory(args: Namespace) -> Module:
	"""
		Instantiate CUDA model from args. Args must be an Namespace containing the attribute "model".
		Available models :
		- VGG11,
		- ResNet18,
		- UBS8KBaseline,
		- WeakBaseline,
		- WeakStrongBaselineRot,
		- dcase2019_model,
	"""
	name = args.model.lower()

	if name == "vgg11":
		model = VGG("VGG11")
	elif name == "resnet18":
		model = ResNet18()
	elif name == "ubs8kbaseline":
		model = UBS8KBaseline()
	elif name == "weakbaseline":
		model = WeakBaselineRot()
	elif name == "weakstrongbaseline":
		model = WeakStrongBaselineRot()
	elif name == "dcase2019":
		model = dcase2019_model()
	else:
		raise RuntimeError("Unknown model \"%s\"" % args.model_name)

	model = model.cuda()
	return model


def optim_factory(args: Namespace, model: Module) -> Optimizer:
	"""
		Instantiate optimizer from args and model.
		Args must be an Namespace containing the attributes "optimizer", "lr" and "weight_decay".
		Available optimizers :
		- Adam,
		- SGD,
	"""
	name = args.optimizer.lower()

	if name == "adam":
		optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif name == "sgd":
		optim = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		raise RuntimeError("Unknown optimizer \"%s\"" % str(args.optim_name))

	return optim


def sched_factory(args: Namespace, optim: Optimizer) -> Optional[object]:
	"""
		Instantiate scheduler from args and optimizer.
		Args must be an Namespace containing the attributes "scheduler", "nb_epochs" and "lr".
		Available optimizers :
		- CosineLRScheduler,
		- None,
	"""
	name = args.scheduler.lower()

	if name in ["cosinelrscheduler", "cosine"]:
		scheduler = CosineLRScheduler(optim, nb_epochs=args.nb_epochs, lr0=args.lr)
	else:
		scheduler = None

	return scheduler


def run_to_train_name(run: str) -> str:
	if run in ["fixmatch", "fm"]:
		return "FixMatch"
	elif run in ["mixmatch", "mm"]:
		return "MixMatch"
	elif run in ["remixmatch", "rmm"]:
		return "ReMixMatch"
	elif run in ["supervised_full", "sf"]:
		return "Supervised_Full"
	elif run in ["supervised_part", "sp"]:
		return "Supervised_Part"
	elif run in ["supervised", "su"]:
		return "Supervised"
	else:
		return ""


def filter_hparams(args: Namespace) -> dict:
	""" Modify hparams values for storing them in SummaryWriter. """
	def filter_item(v):
		if v is None:
			return str(v)
		elif isinstance(v, list):
			return " ".join(v)
		else:
			return v

	hparams = args.__dict__
	for key_, val_ in hparams.items():
		hparams[key_] = filter_item(val_)
	return hparams


def get_nb_parameters(model: Module) -> int:
	return sum(p.numel() for p in model.parameters())


def get_nb_trainable_parameters(model: Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_writer(args: Namespace, start_date: str, pre_suffix: str = "") -> SummaryWriter:
	dirname = (
				  "%s_%s_%s_%s"
				  "%s_%s_%d_%d"
				  "%.2f_%.2f_%.2f_%.2f"
				  "%s_%d_%s_%s"
				  "%s_%s"
			  ) % (
		args.dataset_name, args.model, start_date, args.train_name,
		args.optimizer, args.scheduler, args.batch_size_s, args.batch_size_u,
		args.lambda_u, args.lambda_u1, args.lambda_r, args.threshold_confidence,
		args.use_rampup, args.nb_rampup_epochs, args.criterion_name_u, args.shuffle_s_with_u,
		pre_suffix, args.suffix,
	)

	dirpath = osp.join(args.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=args.train_name)
	return writer


def save_writer(writer: SummaryWriter, args: Namespace, validator: ValidatorABC):
	save_args(writer.log_dir, args)

	"""
	TODO : rem this and validator arg ?
	keys = []
	metrics_recorder = validator.get_metrics_recorder()
	for metrics in validator.get_all_metrics():
		keys += list(metrics.keys())

	metric_dict = {}
	metric_dict.update(
		{"val_max/%s" % name: metrics_recorder.get_max(name) for name in keys})
	metric_dict.update(
		{"val_min/%s" % name: metrics_recorder.get_min(name) for name in keys})
	"""

	writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
	writer.close()


def save_args(filepath: str, args: Namespace):
	with open(filepath, "w") as file:
		json.dump(args.__dict__, file, indent="\t")
