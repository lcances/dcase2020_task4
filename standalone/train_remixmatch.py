import numpy as np
import os.path as osp
import torch

from easydict import EasyDict as edict
from time import time
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomChoice, Compose
from typing import Callable

from dcase2020.augmentation_utils.img_augmentations import Transform
from dcase2020.augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip
from dcase2020.pytorch_metrics.metrics import Metrics

from dcase2020_task4.remixmatch.remixmatch import ReMixMatchMixer, ReMixMatchLoss
from dcase2020_task4.util.MergeDataLoader import MergeDataLoader
from dcase2020_task4.util.rgb_augmentations import Gray, Inversion, RandCrop, Unicolor
from dcase2020_task4.util.confidence_acc import CategoricalConfidenceAccuracy

from .validate import val


def test_remixmatch(
	model: Module, acti_fn: Callable, loader_train_split: MergeDataLoader, loader_val: DataLoader, hparams: edict,
	suffix: str = ""
):
	# ReMixMatch hyperparameters
	hparams.nb_augms_strong = 2  # In paper : 8
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u = 1.5  # In paper : 1.5
	hparams.lambda_u1 = 0.5
	hparams.lambda_r = 0.5
	hparams.lr = 1e-2  # In paper 2e-3
	hparams.weight_decay = 0.02

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
	weak_augm_fn = lambda batch: torch.stack([weak_augm_fn_x(x).cuda() for x in batch]).cuda()
	strong_augm_fn = lambda batch: torch.stack([strong_augm_fn_x(x).cuda() for x in batch]).cuda()

	hparams.train_name = "ReMixMatch"
	dirname = "%s_%s_%s_%s" % (hparams.train_name, hparams.model_name, suffix, hparams.begin_date)
	dirpath = osp.join(hparams.logdir, dirname)
	writer = SummaryWriter(log_dir=dirpath, comment=hparams.train_name)

	optim = SGD(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
	metrics_s = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_val = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_u1 = CategoricalConfidenceAccuracy(hparams.confidence)
	metrics_rot = CategoricalConfidenceAccuracy(hparams.confidence)
	remixmatch = ReMixMatchMixer(
		model, weak_augm_fn, strong_augm_fn, hparams.nb_classes, hparams.nb_augms_strong, hparams.sharpen_temp, hparams.mixup_alpha
	)
	criterion = ReMixMatchLoss(
		lambda_u = hparams.lambda_u,
		lambda_u1 = hparams.lambda_u1,
		lambda_r = hparams.lambda_r,
		mode = "onehot",
	)

	start = time()
	print("\nStart ReMixMatch training ("
		  "%d epochs, %d train examples supervised, %d train examples unsupervised, %d valid examples)..." % (
			  hparams.nb_epochs,
			  len(loader_train_split.loader_supervised) * loader_train_split.loader_supervised.batch_size,
			  len(loader_train_split.loader_unsupervised) * loader_train_split.loader_unsupervised.batch_size,
			  len(loader_val) * loader_val.batch_size
		  ))

	for e in range(hparams.nb_epochs):
		losses, acc_train_s, acc_train_u, acc_train_u1, acc_train_rot = train_remixmatch(
			model, acti_fn, optim, loader_train_split, hparams.nb_classes, criterion, metrics_s, metrics_u, metrics_u1,
			metrics_rot, remixmatch, hparams.lambda_u, hparams.lambda_u1, hparams.lambda_r, e
		)
		acc_val, acc_maxs = val(model, acti_fn, loader_val, hparams.nb_classes, metrics_val, e)

		writer.add_scalar("train/loss", float(np.mean(losses)), e)
		writer.add_scalar("train/acc_s", float(np.mean(acc_train_s)), e)
		writer.add_scalar("train/acc_u", float(np.mean(acc_train_u)), e)
		writer.add_scalar("train/acc_u1", float(np.mean(acc_train_u1)), e)
		writer.add_scalar("train/acc_rot", float(np.mean(acc_train_rot)), e)
		writer.add_scalar("val/acc", float(np.mean(acc_val)), e)
		writer.add_scalar("val/maxs", float(np.mean(acc_maxs)), e)

	print("End ReMixMatch training. (duration = %.2f)" % (time() - start))

	writer.add_hparams(hparam_dict=dict(hparams), metric_dict={})
	writer.close()


def train_remixmatch(
	model: Module,
	acti_fn: Callable,
	optimizer: Optimizer,
	loader: MergeDataLoader,
	nb_classes: int,
	criterion: Callable,
	metrics_s: Metrics,
	metrics_u: Metrics,
	metrics_u1: Metrics,
	metrics_rot: Metrics,
	remixmatch: ReMixMatchMixer,
	lambda_u: float,
	lambda_u1: float,
	lambda_r: float,
	epoch: int,
) -> (list, list, list, list):
	angles_allowed = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
	metrics_s.reset()
	metrics_u.reset()
	metrics_u1.reset()
	metrics_rot.reset()
	train_start = time()
	model.train()

	losses, accuracies_s, accuracies_u, accuracies_u1, accuracies_rot = [], [], [], [], []
	iter_train = iter(loader)
	for i, (batch_s, labels_s, batch_u, _labels_u) in enumerate(iter_train):
		batch_s, batch_u = batch_s.cuda().float(), batch_u.cuda().float()
		labels_s = labels_s.cuda().long()
		labels_s = one_hot(labels_s, nb_classes).float()

		with torch.no_grad():
			remixmatch.distributions.add_batch_pred(labels_s, "labeled")
			remixmatch.distributions.add_batch_pred(torch.softmax(model(batch_u), dim=1), "unlabeled")

		# Apply mix
		batch_x_mixed, labels_x_mixed, batch_u_mixed, labels_u_mixed, batch_u1, labels_u1 = \
			remixmatch(batch_s, labels_s, batch_u)

		# Predict labels for x (mixed), u (mixed) and u1 (strong augment)
		logits_x = model(batch_x_mixed)
		logits_u = model(batch_u_mixed)
		logits_u1 = model(batch_u1)

		# Rotate images and predict rotation for strong augment u1
		batch_u1_rotated, labels_r = apply_random_rot(batch_u1, angles_allowed)
		labels_r = one_hot(labels_r, len(angles_allowed)).float().cuda()
		logits_r = model.forward_rot(batch_u1_rotated)

		# Compute accuracies
		pred_x = acti_fn(logits_x)
		pred_u = acti_fn(logits_u)
		pred_u1 = acti_fn(logits_u1)
		pred_rot = acti_fn(logits_r)

		accuracy_s = metrics_s(pred_x, labels_x_mixed)
		accuracy_u = metrics_u(pred_u, labels_u_mixed)
		accuracy_u1 = metrics_u1(pred_u1, labels_u1)
		accuracy_rot = metrics_rot(pred_rot, labels_r)

		# Update model
		loss = criterion(
			logits_x, labels_x_mixed,
			logits_u, labels_u_mixed,
			logits_u1, labels_u1,
			logits_r, labels_r,
		)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Store data
		losses.append(loss.item())
		accuracies_s.append(metrics_s.value.item())
		accuracies_u.append(metrics_u.value.item())
		accuracies_u1.append(metrics_u1.value.item())
		accuracies_rot.append(metrics_rot.value.item())

		print("Epoch {}, {:d}% \t loss: {:.4e} - sacc: {:.4e} - uacc: {:.4e} - u1acc: {:.4e} - rotacc: {:.4e} - took {:.2f}s".format(
			epoch + 1,
			int(100 * (i + 1) / len(loader)),
			loss.item(),
			accuracy_s,
			accuracy_u,
			accuracy_u1,
			accuracy_rot,
			time() - train_start
		), end="\r")

	print("")
	return losses, accuracies_s, accuracies_u, accuracies_u1, accuracies_rot


def apply_random_rot(batch: Tensor, angles_allowed) -> (Tensor, Tensor):
	idx = np.random.randint(0, len(angles_allowed), len(batch))
	angles = angles_allowed[idx]
	rotate_fn = lambda batch: torch.stack([
		Transform(1.0, rotation=(ang, ang))(x) for x, ang in zip(batch, angles)
	]).cuda()
	res = rotate_fn(batch)
	return res, torch.from_numpy(idx)
