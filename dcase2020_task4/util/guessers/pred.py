from torch import Tensor

from dcase2020_task4.util.guessers.abc import GuesserPredABC
from dcase2020_task4.util.utils_labels import binarize_pred_to_onehot, onehot_to_smooth_onehot, multihot_to_smooth_multihot


class GuesserBinarizeOneHot(GuesserPredABC):
	def __call__(self, pred: Tensor, dim: int) -> Tensor:
		return binarize_pred_to_onehot(pred)


class GuesserThreshold(GuesserPredABC):
	def __init__(self, threshold: float):
		self.threshold = threshold

	def __call__(self, pred: Tensor, dim: int) -> Tensor:
		return (pred > self.threshold).float()


class GuesserSmoothOneHot(GuesserPredABC):
	def __init__(self, smooth: float, nb_classes: int):
		self.smooth = smooth
		self.nb_classes = nb_classes

	def __call__(self, pred: Tensor, dim: int) -> Tensor:
		return onehot_to_smooth_onehot(pred, self.nb_classes, self.smooth)


class GuesserSmoothMultiHot(GuesserPredABC):
	def __init__(self, smooth: float, nb_classes: int):
		self.smooth = smooth
		self.nb_classes = nb_classes

	def __call__(self, pred: Tensor, dim: int) -> Tensor:
		return multihot_to_smooth_multihot(pred, self.nb_classes, self.smooth)
