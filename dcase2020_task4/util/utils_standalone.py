import json
import os.path as osp

from argparse import Namespace
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from dcase2020_task4.validator_abc import ValidatorABC


def filter_hparams(args: Namespace) -> dict:
	""" Modify hparams values for storing them in SummaryWriter. """
	def filter_item(v):
		if v is None:
			return str(v)
		elif isinstance(v, list):
			return " ".join(v)
		else:
			return v

	hparams = dict(args.__dict__)
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
	with open(osp.join(writer.log_dir, "args.json"), "w") as file:
		json.dump(args.__dict__, file, indent="\t")

	keys = []
	metrics_recorder = validator.get_metrics_recorder()
	for metrics in validator.get_all_metrics():
		keys += list(metrics.keys())

	"""
	metric_dict = {}
	metric_dict.update(
		{"val_max/%s" % name: metrics_recorder.get_max(name) for name in keys})
	metric_dict.update(
		{"val_min/%s" % name: metrics_recorder.get_min(name) for name in keys})
	"""

	writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict={})
	writer.close()
