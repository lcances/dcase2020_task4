import torch

from torch import Tensor
from typing import Callable
from metric_utils.metrics import CategoricalAccuracy, Metrics, Recall, Precision


class CategoricalConfidenceAccuracy(CategoricalAccuracy):
	""" Just Categorical Accuracy with a binarization with threshold. """

	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, pred: Tensor, labels: Tensor):
		with torch.no_grad():
			y_pred = (pred > self.confidence).float()
			y_true = (labels > self.confidence).float()
			return super().__call__(y_pred, y_true)


class FnMetric(Metrics):
	def __init__(self, fn: Callable[[Tensor, Tensor], Tensor]):
		super().__init__()
		self.fn = fn

	def __call__(self, pred: Tensor, labels: Tensor):
		super().__call__(pred, labels)

		with torch.no_grad():
			self.value = self.fn(pred, labels).mean()
			self.accumulate_value += self.value

			return self.accumulate_value / self.count


class MaxMetric(FnMetric):
	def __init__(self):
		super().__init__(lambda y_pred, y_true: y_pred.max(dim=1)[0])


class MeanMetric(FnMetric):
	def __init__(self):
		super().__init__(lambda y_pred, y_true: y_pred.mean(dim=1)[0])


class EqConfidenceMetric(Metrics):
	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, pred: Tensor, labels: Tensor):
		super().__call__(pred, labels)

		with torch.no_grad():
			y_pred = (pred > self.confidence).float()
			y_true = (labels > self.confidence).float()

			self.value = (y_pred == y_true).all(dim=1).float().mean()
			self.accumulate_value += self.value
			return self.accumulate_value / self.count


class BinaryConfidenceAccuracy(Metrics):
	def __init__(self, confidence: float, epsilon: float = 1e-10):
		Metrics.__init__(self, epsilon)
		self.confidence = confidence

	def __call__(self, y_pred: Tensor, y_true: Tensor) -> float:
		super().__call__(y_pred, y_true)

		with torch.no_grad():
			y_pred = (y_pred > self.confidence).float()
			correct = (y_pred == y_true).float().sum()
			self.value = correct / (y_true.shape[0] * y_true.shape[1])

			self.accumulate_value += self.value
			return self.accumulate_value / self.count
