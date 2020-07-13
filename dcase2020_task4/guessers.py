import torch

from abc import ABC
from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional

from dcase2020_task4.util.utils_match import binarize_onehot_labels


class GuesserABC(ABC, Callable):
	def __call__(self, x: Tensor, dim: int) -> Tensor:
		raise NotImplementedError("Abstract method")

	def guess_label(self, x: Tensor, dim: int) -> Tensor:
		return self.__call__(x, dim)


class GuesserModelABC(GuesserABC):
	def __call__(self, x: Tensor, dim: int) -> Tensor:
		raise NotImplementedError("Abstract method")

	def get_last_pred(self) -> Optional[Tensor]:
		raise NotImplementedError("Abstract method")


class GuesserModel(GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable):
		self.model = model
		self.acti_fn = acti_fn
		self.last_pred = None

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		with torch.no_grad():
			logits = self.model(batch)
			self.last_pred = self.acti_fn(logits, dim)
			return self.last_pred

	def get_last_pred(self) -> Optional[Tensor]:
		return self.last_pred


class GuesserCompose(GuesserABC):
	def __init__(self, *args):
		self.guessers = list(args)

	def __call__(self, x: Tensor, dim: int) -> Tensor:
		for guesser in self.guessers:
			x = guesser(x, dim)
		return x


# FixMatch guessers
class GuesserOneHot(GuesserABC):
	def __call__(self, pred: Tensor, dim: int) -> Tensor:
		return binarize_onehot_labels(pred)


class GuesserModelOneHot(GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable):
		self.guesser_compose = GuesserCompose(
			GuesserModel(model, acti_fn),
			GuesserOneHot(),
		)

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batch, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


class GuesserThreshold(GuesserABC):
	def __init__(self, threshold: float):
		self.threshold = threshold

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return (batch > self.threshold).float()


class GuesserModelThreshold(GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable, threshold: float):
		self.guesser_compose = GuesserCompose(
			GuesserModel(model, acti_fn),
			GuesserThreshold(threshold)
		)

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batch, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


# MixMatch guessers
class GuesserMeanModel(GuesserABC):
	def __init__(self, model: Module, acti_fn: Callable):
		self.model = model
		self.acti_fn = acti_fn
		self.last_pred = None

	def __call__(self, batches: Tensor, dim: int) -> Tensor:
		nb_augms = batches.shape[0]
		preds = [torch.zeros(0) for _ in range(nb_augms)]
		for k in range(nb_augms):
			logits = self.model(batches[k])
			preds[k] = self.acti_fn(logits, dim=dim)
		preds = torch.stack(preds).cuda()
		self.last_pred = preds
		label_guessed = preds.mean(dim=0)
		return label_guessed

	def get_last_pred(self) -> Optional[Tensor]:
		return self.last_pred


class GuesserMeanModelSharpen(GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable, sharpen_fn: Callable):
		self.guesser_compose = GuesserCompose(
			GuesserMeanModel(model, acti_fn),
			sharpen_fn,
		)

	def __call__(self, batches: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batches, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


# ReMixMatch guessers
class GuesserModelAlignmentSharpen(GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable, avg_distributions: Callable, sharpen_fn: Callable):
		self.guesser_compose = GuesserCompose(
			GuesserModel(model, acti_fn),
			avg_distributions,
			sharpen_fn,
		)

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batch, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()
