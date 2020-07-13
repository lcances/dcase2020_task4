"""
	Methods used in several standalone files for training MixMatch, ReMixMatch or FixMatch.
"""

import inspect
import json
import os.path as osp

from argparse import Namespace

from torch.nn import Module
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from dcase2020_task4.other_models import cnn03
from dcase2020_task4.other_models import resnet
from dcase2020_task4.other_models import ubs8k_baseline
from dcase2020_task4.other_models import vgg
from dcase2020_task4.other_models import weak_baseline_rot
from dcase2020_task4.util.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.validator_abc import ValidatorABC


def check_args(args: Namespace):
	if not osp.isdir(args.dataset_path):
		raise RuntimeError("Invalid dirpath \"%s\"" % args.dataset_path)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath \"%s\"" % args.logdir)
		if not osp.isdir(args.checkpoint_path):
			raise RuntimeError("Invalid dirpath \"%s\"" % args.checkpoint_path)

	if args.dataset_name == "CIFAR10":
		if args.model not in ["VGG11", "ResNet18", "VGG11Rot", "ResNet18Rot", "WideResNet28", "WideResNet28Rot"]:
			raise RuntimeError("Invalid model \"%s\" for dataset \"%s\"" % (args.model, args.dataset_name))
		if args.cross_validation:
			raise RuntimeError("Cross-validation on \"%s\" dataset is not supported." % args.dataset_name)

	elif args.dataset_name == "UBS8K":
		if args.model not in ["UBS8KBaseline", "CNN03", "UBS8KBaselineRot", "CNN03Rot"]:
			raise RuntimeError("Invalid model \"%s\" for dataset \"%s\"" % (args.model, args.dataset_name))
		if not(1 <= args.fold_val <= 10):
			raise RuntimeError("Invalid fold %d (must be in [%d,%d])" % (args.fold_val, 1, 10))


def post_process_args(args: Namespace) -> Namespace:
	if args.args_file is not None:
		args = load_args(args.args_file, args)

	if args.nb_rampup_epochs == "nb_epochs":
		args.nb_rampup_epochs = args.nb_epochs
	args.train_name = run_to_train_name(args.run)
	return args


def get_model_from_name(model_name: str, case_sensitive: bool = False, modules: list = None):
	if modules is None:
		modules = []

	all_members = []
	for module in modules:
		all_members += inspect.getmembers(module)

	for name, obj in all_members:
		if inspect.isclass(obj) or inspect.isfunction(obj):
			obj_name = obj.__name__
			if obj_name == model_name or (not case_sensitive and obj_name.lower() == model_name.lower()):
				return obj

	raise AttributeError("This model does not exist: %s " % model_name)


def model_factory(args: Namespace, case_sensitive: bool = False, modules: list = None) -> Module:
	"""
		Instantiate CUDA model from args. Args must be an Namespace containing the attribute "model".
		Models available are in files : cnn03, resnet, ubs8k_baseline, vgg, weak_baseline_rot
			(in directory "dcase2020_task4/other_models/").
	"""
	if modules is None:
		modules = []
	modules += [cnn03, resnet, ubs8k_baseline, vgg, weak_baseline_rot]

	model_class = get_model_from_name(args.model, case_sensitive, modules)
	model = model_class().cuda()

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
		raise RuntimeError("Unknown optimizer \"%s\"" % str(args.optimizer))

	return optim


def sched_factory(args: Namespace, optim: Optimizer) -> Optional[object]:
	"""
		Instantiate scheduler from args and optimizer.
		Args must be an Namespace containing the attributes "scheduler", "nb_epochs" and "lr".
		Available optimizers :
		- CosineLRScheduler,
		- None,
	"""
	name = str(args.scheduler).lower()

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
	dirname = ""
	dirname += "%s_%s_%s_%s" % (
		args.dataset_name, start_date, args.model, args.train_name)
	dirname += "%d_%d_%s_%.2e" % (
		args.batch_size_s, args.batch_size_u, args.optimizer, args.lr)
	dirname += "%s_%.2e_%.2e_%.2e" % (
		args.scheduler, args.lambda_u, args.lambda_u1, args.lambda_r)
	dirname += "%s_%d_%s_%.2e" % (
		args.use_rampup, args.nb_rampup_epochs, args.shuffle_s_with_u, args.threshold_confidence)
	dirname += "%s" % (
		args.criterion_name_u)

	dirname += "%s_%s" % (pre_suffix, args.suffix)
	dirpath = osp.join(args.logdir, dirname)

	writer = SummaryWriter(log_dir=dirpath, comment=args.train_name)
	return writer


def save_writer(writer: SummaryWriter, args: Namespace, validator: ValidatorABC):
	filepath = osp.join(writer.log_dir, "args.json")
	save_args(filepath, args)

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
	print("Data will saved in tensorboard writer \"%s\"." % writer.log_dir)


def save_args(filepath: str, args: Namespace):
	with open(filepath, "w") as file:
		json.dump(args.__dict__, file, indent="\t")
	print("Arguments saved in file \"%s\"." % filepath)


def load_args(filepath: str, args: Namespace, check_keys: bool = True) -> Namespace:
	with open(filepath, "r") as file:
		args_dict = json.load(file)

		if check_keys:
			differences = set(args_dict.keys()).difference(args.__dict__.keys())
			if len(differences) > 0:
				raise RuntimeError("Found unknown(s) key(s) in JSON file : \"%s\"." % ", ".join(differences))

		args.__dict__.update(args_dict)

	return args
