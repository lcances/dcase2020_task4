import json
import os.path as osp
import torch

from argparse import Namespace
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import List, Union

from dcase2020_task4.validator_abc import ValidatorABC


def normalized(batch: Tensor, dim: int) -> Tensor:
	""" Return the vector normalized. """
	return batch / batch.norm(p=1, dim=dim, keepdim=True)


def same_shuffle(values: List[Tensor]) -> List[Tensor]:
	""" Shuffle each value of values with the same indexes. """
	indices = torch.randperm(len(values[0]))
	for i in range(len(values)):
		values[i] = values[i][indices]
	return values


def binarize_onehot_labels(batch: Tensor) -> Tensor:
	""" Convert list of distributions vectors to one-hot. """
	indexes = batch.argmax(dim=1)
	nb_classes = batch.shape[1]
	bin_labels = one_hot(indexes, nb_classes)
	return bin_labels


def label_to_num(one_hot_vectors: Tensor):
	""" Convert a list of one-hot vectors of size (N, C) to a list of classes numbers of size (N). """
	return one_hot_vectors.argmax(dim=1)


def merge_first_dimension(t: Tensor) -> Tensor:
	""" Reshape tensor of size (M, N, ...) to (M*N, ...). """
	shape = list(t.size())
	if len(shape) < 2:
		raise RuntimeError("Invalid nb of dimension (%d) for merge_first_dimension. Should have at least 2 dimensions." % len(shape))
	return t.reshape(shape[0] * shape[1], *shape[2:])


def cross_entropy_with_logits(logits: Tensor, targets: Tensor, dim: Union[int, tuple] = 1) -> Tensor:
	"""
		Apply softmax on logits and compute cross-entropy with targets.
		Target must be a (batch_size, nb_classes) tensor.
	"""
	pred_x = torch.softmax(logits, dim=dim)
	return cross_entropy(pred_x, targets, dim)


def cross_entropy(pred: Tensor, targets: Tensor, dim: Union[int, tuple] = 1) -> Tensor:
	"""
		Compute cross-entropy with targets.
		Target must be a (batch_size, nb_classes) tensor.
	"""
	return -torch.sum(torch.log(pred) * targets, dim=dim)


def get_lrs(optim: Optimizer) -> List[float]:
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	return get_lrs(optim)[idx]


def set_lr(optim: Optimizer, new_lr: float):
	for group in optim.param_groups:
		group["lr"] = new_lr


def build_writer(args: Namespace, suffix: str = "") -> SummaryWriter:
	dirname = "%s_%s_%s_%s_%s" % (args.dataset_name, args.train_name, args.model_name, args.begin_date, suffix)
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
	metric_dict = {}
	metric_dict.update(
		{"val_max/%s" % name: metrics_recorder.get_max(name) for name in keys})
	metric_dict.update(
		{"val_min/%s" % name: metrics_recorder.get_min(name) for name in keys})

	writer.add_hparams(hparam_dict=filter_hparams(args), metric_dict=metric_dict)
	writer.close()


def multi_hot(labels_nums: List[List[int]], nb_classes: int) -> Tensor:
	""" TODO : test this fn """
	res = torch.zeros((len(labels_nums), nb_classes))
	for i, nums in enumerate(labels_nums):
		res[i] = torch.sum(torch.stack([one_hot(num) for num in nums]), dim=0)
	return res


def multilabel_to_num(labels: Tensor) -> List[List[int]]:
	""" TODO : test this fn """
	res = [[] for _ in range(len(labels))]
	for i, label in enumerate(labels):
		for j, bin in enumerate(label):
			if bin == 1.0:
				res[i].append(j)
	return res


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
