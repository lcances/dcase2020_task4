import numpy as np
import os.path as osp
import torch

from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomChoice, Compose

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip
from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.fixmatch.fixmatch import fixmatch_loss
from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.rgb_augmentations import Gray, Inversion, RandCrop, Unicolor
from dcase2020_task4.util.match_utils import binarize_labels
from dcase2020_task4.util.confidence_acc import CategoricalConfidenceAccuracy

from .validate import val


def test_fixmatch(model: Module, loader_train_split: MergeDataLoader, loader_val: DataLoader, hparams: edict,
				  suffix: str = ""):
	# FixMatch hyperparameters
	hparams.lambda_u = 1.0
	hparams.lr0 = 0.03  # learning rate, eta
	hparams.beta = 0.9  # used only for SGD
	hparams.threshold = 0.95  # tau
	hparams.batch_size = 16  # in paper: 64
	hparams.weight_decay = 1e-4

	weak_augm_fn_x = RandomChoice([
		HorizontalFlip(0.5),
		VerticalFlip(0.5),
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
	])
	strong_augm_fn_x = Compose([
		RandomChoice([
			Transform(1.0, scale=(0.5, 1.5)),
			Transform(1.0, rotation=(-np.pi, np.pi)),
		]),
		RandomChoice([
			Gray(1.0),
			RandCrop(1.0),
			Unicolor(1.0),
			Inversion(1.0),
		]),
	])
	weak_augm_fn = lambda batch: torch.stack([weak_augm_fn_x(x).cuda() for x in batch])
	strong_augm_fn = lambda batch: torch.stack([strong_augm_fn_x(x).cuda() for x in batch])

	hparams.train_name = "FixMatch"
	dirname = "%s_%s_%s_%s" % (hparams.train_name, hparams.model_name, suffix, hparams.begin_date)
	dirpath = osp.join(hparams.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=hparams.train_name)

	optim = SGD(model.parameters(), lr=hparams.lr0, weight_decay=hparams.weight_decay)
	metrics_s = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_val = CategoricalConfidenceAccuracy(hparams.confidence)

	start = time()
	print("\nStart FixMatch training ("
		  "%d epochs, %d train examples supervised, %d train examples unsupervised, %d valid examples)..." % (
			  hparams.nb_epochs,
			  len(loader_train_split.loader_supervised) * loader_train_split.loader_supervised.batch_size,
			  len(loader_train_split.loader_unsupervised) * loader_train_split.loader_unsupervised.batch_size,
			  len(loader_val) * loader_val.batch_size
		  ))

	for e in range(hparams.nb_epochs):
		losses, acc_train_s, acc_train_u = train_fixmatch(
			model, optim, loader_train_split, hparams.nb_classes, strong_augm_fn, weak_augm_fn, metrics_s, metrics_u,
			hparams.threshold, hparams.lambda_u, e
		)
		acc_val = val(model, loader_val, hparams.nb_classes, metrics_val, e)

		optim.lr = hparams.lr0 * np.cos(7.0 * np.pi * e / (16.0 * hparams.nb_epochs))

		writer.add_scalar("train/loss", float(np.mean(losses)), e)
		writer.add_scalar("train/acc_s", float(np.mean(acc_train_s)), e)
		writer.add_scalar("train/acc_u", float(np.mean(acc_train_u)), e)
		writer.add_scalar("train/lr", optim.lr, e)
		writer.add_scalar("val/acc", float(np.mean(acc_val)), e)

	print("End FixMatch training. (duration = %.2f)" % (time() - start))

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def train_fixmatch(
	model: Module,
	optimizer: Optimizer,
	loader: MergeDataLoader,
	nb_classes: int,
	strong_augm_fn,
	weak_augm_fn,
	metrics_s: Metrics,
	metrics_u: Metrics,
	threshold: float,
	lambda_u: float,
	epoch: int,
) -> (list, list, list):
	metrics_s.reset()
	metrics_u.reset()
	train_start = time()
	model.train()

	losses, accuracies_s, accuracies_u = [], [], []
	iter_train = iter(loader)
	for i, (batch_s, labels_s, batch_u, _labels_u) in enumerate(iter_train):
		batch_s, batch_u = batch_s.cuda().float(), batch_u.cuda().float()
		labels_s = labels_s.cuda().long()
		labels_s = one_hot(labels_s, nb_classes).float()

		# Apply augmentations
		batch_s_weak = weak_augm_fn(batch_s).cuda()
		batch_u_weak = weak_augm_fn(batch_u).cuda()
		batch_u_strong = strong_augm_fn(batch_u).cuda()

		# Compute logits
		logits_s_weak = model(batch_s_weak)
		logits_u_weak = model(batch_u_weak)
		logits_u_strong = model(batch_u_strong)

		# Compute accuracies
		pred_s_weak = torch.softmax(logits_s_weak, dim=1)
		pred_u_strong = torch.softmax(logits_u_strong, dim=1)
		label_u_guessed = binarize_labels(logits_u_weak)

		accuracy_s = metrics_s(pred_s_weak, labels_s)
		accuracy_u = metrics_u(pred_u_strong, label_u_guessed)

		# Update model
		loss = fixmatch_loss(logits_s_weak, labels_s, logits_u_weak, logits_u_strong, threshold, lambda_u)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Store data
		losses.append(loss.item())
		accuracies_s.append(metrics_s.value.item())
		accuracies_u.append(metrics_u.value.item())

		print("Epoch {}, {:d}% \t loss: {:.4e} - sacc: {:.4e} - uacc: {:.4e} - took {:.2f}s".format(
			epoch + 1,
			int(100 * (i + 1) / len(loader)),
			loss.item(),
			accuracy_s,
			accuracy_u,
			time() - train_start
		), end="\r")

	print("")
	return losses, accuracies_s, accuracies_u
