import torch

from abc import ABC
from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional

from dcase2020_task4.util.guessers.abc import GuesserModelABC, GuesserCompose
from dcase2020_task4.util.guessers.pred import GuesserBinarizeOneHot, GuesserThreshold, GuesserSmoothOneHot, GuesserSmoothMultiHot


class GuesserModelOther(ABC, GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable, *args):
		self.guesser_compose = GuesserCompose(
			GuesserModel(model, acti_fn),
			*args
		)

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batch, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


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


# FixMatch guessers
class GuesserModelBinarize(GuesserModelOther):
	def __init__(self, model: Module, acti_fn: Callable):
		super().__init__(model, acti_fn, GuesserBinarizeOneHot())


class GuesserModelThreshold(GuesserModelOther):
	def __init__(self, model: Module, acti_fn: Callable, threshold: float):
		super().__init__(model, acti_fn, GuesserThreshold(threshold))


class GuesserModelBinarizeSmooth(GuesserModelOther):
	def __init__(self, model: Module, acti_fn: Callable, smooth: float, nb_classes: int):
		super().__init__(model, acti_fn, GuesserBinarizeOneHot(), GuesserSmoothOneHot(smooth, nb_classes))


class GuesserModelThresholdSmooth(GuesserModelOther):
	def __init__(self, model: Module, acti_fn: Callable, threshold: float, smooth: float, nb_classes: int):
		super().__init__(model, acti_fn, GuesserThreshold(threshold), GuesserSmoothMultiHot(smooth, nb_classes))


# MixMatch guessers
class GuesserModelSharpen(GuesserModelOther):
	def __init__(self, model: Module, acti_fn: Callable, sharpen_fn: Callable):
		super().__init__(model, acti_fn, sharpen_fn)


class GuesserMeanModelBinarize(GuesserModelABC):
	def __init__(self, model: Module, acti_fn: Callable):
		self.guesser_compose = GuesserCompose(
			GuesserMeanModel(model, acti_fn),
			GuesserBinarizeOneHot(),
		)

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batch, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


class GuesserMeanModel(GuesserModelABC):
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
		label_guessed = preds.mean(dim=0)
		self.last_pred = label_guessed
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
class GuesserModelAlignmentSharpen(GuesserModelOther):
	def __init__(self, model: Module, acti_fn: Callable, avg_distributions: Callable, sharpen_fn: Callable):
		super().__init__(model, acti_fn, avg_distributions, sharpen_fn)
