"""
	Methods used in several standalone files for training MixMatch, ReMixMatch or FixMatch.
"""

import inspect
import json
import os.path as osp
import subprocess

from argparse import Namespace

from torch.nn import Module
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Optional, Union

from dcase2020_task4.other_models import cnn03, cnn03mish, resnet, ubs8k_baseline, vgg, weak_baseline_rot, wrn28_2, wrn_old
from dcase2020_task4.util.cosine_scheduler import CosineLRScheduler
from dcase2020_task4.util.radam import RAdam, PlainRAdam, AdamW
from dcase2020_task4.util.step_scheduler import StepLRScheduler


FLOAT_FORMAT = "%.3f"


# Full train names constants
class FULLNAME:
	FIXMATCH = "FixMatch"
	MIXMATCH = "MixMatch"
	REMIXMATCH = "ReMixMatch"
	SUPERVISED_FULL = "Supervised_Full"
	SUPERVISED_PART = "Supervised_Part"
	SUPERVISED = "Supervised"


def check_args(args: Namespace):
	"""
		Check arguments (mainly directories and files)
		@param args: argparse arguments.
	"""
	if not osp.isdir(args.dataset_path):
		raise RuntimeError("Invalid dirpath \"%s\"" % args.dataset_path)

	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath \"%s\"" % args.logdir)
		if not osp.isdir(args.checkpoint_path):
			raise RuntimeError("Invalid dirpath \"%s\"" % args.checkpoint_path)

	if args.dataset_name == "CIFAR10":
		if args.model not in ["VGG11", "VGG11Rot",
							  "ResNet18", "ResNet18Rot",
							  "WideResNet", "WideResNetRot",
							  "WideResNet28", "WideResNet28Rot",
							  "WideResNetOld", "WideResNetRotOld"]:
			raise RuntimeError("Invalid model \"%s\" for dataset \"%s\"" % (args.model, args.dataset_name))
		if args.cross_validation:
			raise RuntimeError("Cross-validation on \"%s\" dataset is not supported." % args.dataset_name)

	elif args.dataset_name == "UBS8K":
		if args.model not in ["UBS8KBaseline", "UBS8KBaselineRot", "CNN03", "CNN03Rot"]:
			raise RuntimeError("Invalid model \"%s\" for dataset \"%s\"" % (args.model, args.dataset_name))
		if args.fold_val is not None and not(1 <= args.fold_val <= 10):
			raise RuntimeError("Invalid fold \"%d\" (must be in [%d,%d])" % (args.fold_val, 1, 10))

	if args.args_filepaths is not None:
		for filepath in args.args_filepaths:
			if not osp.isfile(filepath):
				raise RuntimeError("Invalid filepath \"%s\"." % filepath)


def post_process_args(args: Namespace) -> Namespace:
	"""
		Update arguments by adding some parameters inside.
		@param args: argparse arguments.
		@return: The updated argparse arguments.
	"""
	args.train_name = None
	args.git_hash = None
	args_filepaths = args.args_filepaths

	if args.args_filepaths is not None:
		for filepath in args.args_filepaths:
			args = load_args(filepath, args)

	if args.nb_rampup_steps == "nb_epochs":
		args.nb_rampup_steps = args.nb_epochs

	args.train_name = get_full_train_name(args.run)
	args.git_hash = get_current_git_hash()
	args.args_filepaths = args_filepaths

	return args


def get_current_git_hash() -> str:
	"""
		Return the current git hash in the current directory.
		@return: The git hash.
	"""
	try:
		git_hash = subprocess.check_output(["git", "describe", "--always"])
		git_hash = git_hash.decode("UTF-8").replace("\n", "")
		return git_hash
	except subprocess.CalledProcessError:
		return "UNKNOWN"


def get_model_from_name(model_name: str, case_sensitive: bool = False, modules: list = None) -> Callable:
	"""
		Return a model from models available in several modules.
		@param model_name: The name of the model to get.
		@param case_sensitive: Use case sensitive check for model name with model available.
		@param modules: The python modules where to search models classes or functions.
		@return: The class or function of the model selected.
	"""
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

	raise AttributeError("This model does not exist: \"%s\" " % model_name)


def build_model_from_args(args: Namespace, case_sensitive: bool = False, modules: list = None) -> Module:
	"""
		Instantiate CUDA model from args. Args must be an Namespace containing the attribute "model".
		Models available are in files : cnn03, cnn03mish, resnet, ubs8k_baseline, vgg, weak_baseline_rot
			(in directory "dcase2020_task4/other_models/").
	"""
	if modules is None:
		modules = []
	modules += [cnn03, cnn03mish, resnet, ubs8k_baseline, vgg, weak_baseline_rot, wrn28_2, wrn_old]

	model_class = get_model_from_name(args.model, case_sensitive, modules)

	if hasattr(model_class, "from_args"):
		model = model_class.from_args(args)
	else:
		model = model_class()

	model = model.cuda()
	return model


def build_optim_from_args(args: Namespace, model: Module) -> Optimizer:
	"""
		Instantiate optimizer from args and model.
		Args must be an Namespace containing the attributes "optimizer", "lr" and optionals "weight_decay" and "momentum".
	"""
	name = args.optimizer.lower()

	if args.weight_decay is None:
		kwargs = dict(params=model.parameters(), lr=args.lr)
	else:
		kwargs = dict(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	if name == "adam":
		optim = Adam(**kwargs)
	elif name == "sgd":
		if args.momentum is None:
			optim = SGD(**kwargs)
		else:
			optim = SGD(**kwargs, momentum=args.momentum)
	elif name == "radam":
		optim = RAdam(**kwargs)
	elif name == "plainradam":
		optim = PlainRAdam(**kwargs)
	elif name == "adamw":
		optim = AdamW(**kwargs)
	else:
		raise RuntimeError("Unknown optimizer \"%s\"." % str(args.optimizer))

	return optim


def build_sched_from_args(args: Namespace, optim: Optimizer) -> Optional[object]:
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
	elif name in ["steplrscheduler", "step"]:
		# TODO : rem ?
		print("WARNING: Depreciated scheduler. Use MultiStepLR instead.")
		scheduler = StepLRScheduler(optim, lr0=args.lr, lr_decay_ratio=args.lr_decay_ratio, epoch_steps=args.epoch_steps)
	elif name in ["multisteplr"]:
		scheduler = MultiStepLR(optim, milestones=args.epoch_steps, gamma=args.lr_decay_ratio)
	else:
		scheduler = None

	return scheduler


def get_full_train_name(run: str) -> str:
	"""
		Returns full train name from a acronym.
		@param run: The acronym of the train chosen by the user.
		@return: The full name.
	"""
	if run in ["fixmatch", "fm"]:
		full_name = FULLNAME.FIXMATCH
	elif run in ["mixmatch", "mm"]:
		full_name = FULLNAME.MIXMATCH
	elif run in ["remixmatch", "rmm"]:
		full_name = FULLNAME.REMIXMATCH
	elif run in ["supervised_full", "sf"]:
		full_name = FULLNAME.SUPERVISED_FULL
	elif run in ["supervised_part", "sp"]:
		full_name = FULLNAME.SUPERVISED_PART
	elif run in ["supervised", "su"]:
		full_name = FULLNAME.SUPERVISED
	else:
		full_name = ""
	return full_name


def get_nb_parameters(model: Module) -> int:
	"""
		Return the number of parameters in a model.
		@param model: Pytorch Module to check.
		@return: The number of parameters.
	"""
	return sum(p.numel() for p in model.parameters())


def get_nb_trainable_parameters(model: Module) -> int:
	"""
		Return the number of trainable parameters in a model.
		@param model: Pytorch Module.
		@return: The number of trainable parameters.
	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_writer_from_args(args: Namespace, start_date: str, augments: Any, pre_suffix: str = "") -> SummaryWriter:
	"""
		Build the tensorboard writer object with a specific name and directory.
		@param args: argparse arguments.
		@param start_date: Start date of the program.
		@param augments: A dictionary or list of augments used.
		@param pre_suffix: An additional suffix added before the real suffix given by "args.suffix".
		@return: The SummaryWriter object created for saving data.
	"""
	dirname = "%s_%s_%s_%s_%s_%s" % (args.dataset_name, start_date, args.model, args.train_name, pre_suffix, args.suffix)
	dirpath = osp.join(args.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=args.train_name)

	def filter_args(args_: Namespace) -> dict:
		""" Modify args values for storing them in SummaryWriter. """

		def filter_item(v):
			if v is None:
				return str(v)
			elif isinstance(v, list):
				return " ".join(v)
			else:
				return v

		hparams = args_.__dict__
		for key_, val_ in hparams.items():
			hparams[key_] = filter_item(val_)
		return hparams

	writer.add_hparams(hparam_dict=filter_args(args), metric_dict={})
	writer.add_text("args", json.dumps(args.__dict__, indent="\t"))
	writer.add_text("augments", json.dumps(to_dict_rec(augments), indent="\t"))

	return writer


def save_args(filepath: str, args: Namespace):
	"""
		Save arguments in JSON file.
		@param filepath: The filepath where to save the arguments.
		@param args: argparse arguments.
	"""
	content = {"args": args.__dict__}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")
	print("Arguments saved in file \"%s\"." % filepath)


def load_args(filepath: str, args: Namespace, check_keys: bool = True) -> Namespace:
	"""
		Load arguments from a JSON file.
		@param filepath: The path to JSON file.
		@param args: argparse arguments to update.
		@param check_keys: If True, check if keys of JSON file are inside args keys.
		@return: The argparse arguments updated.
	"""
	with open(filepath, "r") as file:
		file_dict = json.load(file)
		if "args" not in file_dict.keys():
			raise RuntimeError("Invalid args file or too old args file version.")

		args_file_dict = file_dict["args"]

		if check_keys:
			differences = set(args_file_dict.keys()).difference(args.__dict__.keys())
			if len(differences) > 0:
				raise RuntimeError("Found unknown(s) key(s) in JSON file : \"%s\"." % ", ".join(differences))

		args.__dict__.update(args_file_dict)

		# Post process : convert "none" strings to None value
		for name, value in args.__dict__.items():
			if isinstance(value, str) and value.lower() == "none":
				args.__dict__[name] = None

	return args


def save_augms(filepath: str, augms: Any):
	"""
		Save augments to JSON file.
		@param filepath: The path to JSON file.
		@param augms: A dictionary or list of augments used.
	"""
	content = {"augments": to_dict_rec(augms, "__class__")}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")
	print("Augments names saved in file \"%s\"." % filepath)


def to_dict_rec(obj: Any, class_name_key: Optional[str] = "__class__") -> Union[dict, list]:
	"""
		Convert variable to dictionary.
		@param obj: The object to convert.
		@param class_name_key: Key used to save the class name if we convert an object.
		@return:
	"""
	# Code imported from (with small changes) :
	# https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

	if isinstance(obj, dict):
		data = {}
		for key, value in obj.items():
			data[key] = to_dict_rec(value, class_name_key)
		return data
	elif hasattr(obj, "_ast"):
		return to_dict_rec(obj._ast())
	elif hasattr(obj, "__iter__") and not isinstance(obj, str):
		return [to_dict_rec(v, class_name_key) for v in obj]
	elif hasattr(obj, "__dict__"):
		data = {}
		if class_name_key is not None and hasattr(obj, "__class__"):
			data[class_name_key] = obj.__class__.__name__
		data.update(dict([
			(key, to_dict_rec(value, class_name_key))
			for key, value in obj.__dict__.items()
			if not callable(value) and not key.startswith('_')
		]))
		return data
	else:
		return obj
