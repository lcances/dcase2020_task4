import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional

from dcase2020_task4.util.guessers.abc import GuesserModelABC, GuesserCompose
from dcase2020_task4.util.guessers.pred import GuesserArgmax, GuesserThreshold, GuesserSmoothOneHot, GuesserSmoothMultiHot


class GuesserModelOther(GuesserModelABC):
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
class GuesserModelArgmax(GuesserModelOther):
	"""
		Use model to compute prediction and binarize with "one_hot(argmax(.))".
	"""
	def __init__(self, model: Module, acti_fn: Callable):
		super().__init__(model, acti_fn, GuesserArgmax())


class GuesserModelThreshold(GuesserModelOther):
	"""
		Use model to compute prediction and binarize with values above a threshold.
	"""
	def __init__(self, model: Module, acti_fn: Callable, threshold: float):
		super().__init__(model, acti_fn, GuesserThreshold(threshold))


class GuesserModelArgmaxSmooth(GuesserModelOther):
	"""
		Use model to compute prediction, binarize with "one_hot(argmax(.))" and smooth label.
	"""
	def __init__(self, model: Module, acti_fn: Callable, smooth: float, nb_classes: int):
		super().__init__(model, acti_fn, GuesserArgmax(), GuesserSmoothOneHot(smooth, nb_classes))


class GuesserModelThresholdSmooth(GuesserModelOther):
	"""
		Use model to compute prediction, binarize with values above a threshold and smooth label.
	"""
	def __init__(self, model: Module, acti_fn: Callable, threshold: float, smooth: float, nb_classes: int):
		super().__init__(model, acti_fn, GuesserThreshold(threshold), GuesserSmoothMultiHot(smooth, nb_classes))


# MixMatch guessers
class GuesserModelSharpen(GuesserModelOther):
	"""
		Use model to compute prediction and apply sharpening function on it.
	"""
	def __init__(self, model: Module, acti_fn: Callable, sharpen_fn: Callable):
		super().__init__(model, acti_fn, sharpen_fn)


class GuesserMeanModel(GuesserModelABC):
	"""
		Use model to compute predictions on several batches and compute mean of predictions.
	"""
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
	"""
		Use model to compute predictions on several batches, compute mean of predictions and apply sharpening function to compute final label.
	"""
	def __init__(self, model: Module, acti_fn: Callable, sharpen_fn: Callable):
		self.guesser_compose = GuesserCompose(
			GuesserMeanModel(model, acti_fn),
			sharpen_fn,
		)

	def __call__(self, batches: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batches, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


class GuesserMeanModelArgmax(GuesserModelABC):
	"""
		Use model to compute predictions on several batches, compute mean of predictions and apply "onehot(argmax(.))" to compute final label.
	"""
	def __init__(self, model: Module, acti_fn: Callable):
		self.guesser_compose = GuesserCompose(
			GuesserMeanModel(model, acti_fn),
			GuesserArgmax(),
		)

	def __call__(self, batch: Tensor, dim: int) -> Tensor:
		return self.guesser_compose(batch, dim)

	def get_last_pred(self) -> Optional[Tensor]:
		return self.guesser_compose.guessers[0].get_last_pred()


# ReMixMatch guessers
class GuesserModelAlignmentSharpen(GuesserModelOther):
	"""
		Use model to compute prediction, apply distribution alignment and sharpening function on it.
	"""
	def __init__(self, model: Module, acti_fn: Callable, avg_distributions: Callable, sharpen_fn: Callable):
		super().__init__(model, acti_fn, avg_distributions, sharpen_fn)
