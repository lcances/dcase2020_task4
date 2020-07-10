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

from dcase2020_task4.dcase2019.models import dcase2019_model
from dcase2020_task4.other_models.weak_baseline_rot import WeakBaselineRot, WeakStrongBaselineRot
from dcase2020_task4.other_models.vgg import VGG
from dcase2020_task4.other_models.resnet import ResNet18
from dcase2020_task4.other_models.UBS8KBaseline import UBS8KBaseline
from dcase2020_task4.validator_abc import ValidatorABC


def model_factory(args: Namespace) -> Module:
	"""
		Instantiate CUDA model from args. Args must be an Namespace containing the attribute "model_name".
		Available models :
		- VGG11,
		- ResNet18,
		- UBS8KBaseline,
		- WeakBaseline,
		- WeakStrongBaselineRot,
		- dcase2019_model,
	"""
	name = args.model_name.lower()

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
		Instantiate optimizer from args.
		Args must be an Namespace containing the attributes "optim_name", "lr" and "weight_decay".
		Available optimizers :
		- Adam,
		- SGD,
	"""
	name = args.optim_name.lower()

	if name == "adam":
		optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif name == "sgd":
		optim = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		raise RuntimeError("Unknown optimizer \"%s\"" % str(args.optim_name))

	return optim


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


def build_writer(args: Namespace, start_date: str, suffix: str = "") -> SummaryWriter:
	dirname = "%s_%s_%s_%s_%s" % (args.dataset_name, args.train_name, args.model_name, start_date, suffix)
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
