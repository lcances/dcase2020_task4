import numpy as np
import os.path as osp
import torch

from easydict import EasyDict as edict
from time import time
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomChoice

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip
from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.mixmatch.mixmatch import mixmatch_loss, MixMatch
from dcase2020_task4.mixmatch.RampUp import RampUp
from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.rgb_augmentations import Gray, Inversion, RandCrop, Unicolor
from dcase2020_task4.util.confidence_acc import CategoricalConfidenceAccuracy

from .validate import val


def test_mixmatch(model: Module, loader_train_split: MergeDataLoader, loader_val: DataLoader, hparams: edict, suffix: str = ""):
	# MixMatch hyperparameters
	hparams.nb_augms = 2
	hparams.sharpen_val = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u_max = 75.0
	hparams.lr = 1e-2
	hparams.weight_decay = 0.0008

	nb_rampup_steps = hparams.nb_epochs * len(loader_train_split)
	augment_fn_x = RandomChoice([
		HorizontalFlip(0.5),
		VerticalFlip(0.5),
		Transform(0.5, scale=(0.75, 1.25)),
		Transform(0.5, rotation=(-np.pi, np.pi)),
		Gray(0.5),
		RandCrop(0.5, rect_max_scale=(0.2, 0.2)),
		Unicolor(0.5),
		Inversion(0.5),
	])
	nb_classes = hparams.nb_classes
	confidence = hparams.confidence

	hparams.train_name = "MixMatch"
	dirname = "%s_%s_%s_%s" % (hparams.train_name, hparams.model_name, suffix, hparams.begin_date)
	dirpath = osp.join(hparams.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=hparams.train_name)

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
	metrics_s = CategoricalConfidenceAccuracy(confidence)
	metrics_u = CategoricalConfidenceAccuracy(confidence)
	metrics_val = CategoricalConfidenceAccuracy(confidence)
	mixmatch = MixMatch(model, augment_fn_x, hparams.nb_augms, hparams.sharpen_val, hparams.mixup_alpha)
	lambda_u_rampup = RampUp(max_value=hparams.lambda_u_max, nb_steps=nb_rampup_steps)

	start = time()
	print("\nStart MixMatch training ("
		  "%d epochs, %d train examples supervised, %d train examples unsupervised, %d valid examples)..." % (
			  hparams.nb_epochs,
			  len(loader_train_split.loader_supervised) * loader_train_split.loader_supervised.batch_size,
			  len(loader_train_split.loader_unsupervised) * loader_train_split.loader_unsupervised.batch_size,
			  len(loader_val) * loader_val.batch_size
		  ))

	for e in range(hparams.nb_epochs):
		losses, acc_train_s, acc_train_u = train_mixmatch(
			model, optim, loader_train_split, nb_classes, metrics_s, metrics_u, mixmatch, lambda_u_rampup, e
		)
		acc_val = val(model, loader_val, nb_classes, metrics_val, e)

		writer.add_scalar("train/loss", float(np.mean(losses)), e)
		writer.add_scalar("train/acc_s", float(np.mean(acc_train_s)), e)
		writer.add_scalar("train/acc_u", float(np.mean(acc_train_u)), e)
		writer.add_scalar("val/acc", float(np.mean(acc_val)), e)

	print("End MixMatch training. (duration = %.2f)" % (time() - start))

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def train_mixmatch(
	model: Module,
	optimizer: Optimizer,
	loader: MergeDataLoader,
	nb_classes: int,
	metrics_s: Metrics,
	metrics_u: Metrics,
	mixmatch: MixMatch,
	lambda_u_rampup: RampUp,
	epoch: int
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

		# Apply mix
		batch_x_mixed, labels_x_mixed, batch_u_mixed, labels_u_mixed = mixmatch(batch_s, labels_s, batch_u)

		# Compute logits
		logits_x = model(batch_x_mixed)
		logits_u = model(batch_u_mixed)

		# Compute accuracies
		pred_x = torch.softmax(logits_x, dim=1)
		pred_u = torch.softmax(logits_u, dim=1)

		accuracy_s = metrics_s(pred_x, labels_x_mixed)
		accuracy_u = metrics_u(pred_u, labels_u_mixed)

		# Update model
		loss = mixmatch_loss(logits_x, labels_x_mixed, logits_u, labels_u_mixed, lambda_u_rampup.value)
		optimizer.zero_grad()
		loss.backward()
		clip_grad_norm_(model.parameters(), 100)
		optimizer.step()

		lambda_u_rampup.step()

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
