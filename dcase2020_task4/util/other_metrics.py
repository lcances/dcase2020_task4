
from torch import Tensor
from typing import Callable
from dcase2020_task4.pytorch_metrics.metrics import CategoricalAccuracy, Metrics


class CategoricalConfidenceAccuracy(CategoricalAccuracy):
	""" Just Categorical Accuracy with a binarization with threshold. """

	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, pred: Tensor, labels: Tensor):
		y_pred = (pred > self.confidence).float()
		y_true = (labels > self.confidence).float()
		return super().__call__(y_pred, y_true)


class FnMetric(Metrics):
	def __init__(self, fn: Callable[[Tensor, Tensor], Tensor]):
		super().__init__()
		self.fn = fn

	def __call__(self, pred: Tensor, labels: Tensor):
		super().__call__(pred, labels)

		self.value = self.fn(pred, labels).mean()
		self.accumulate_value += self.value

		return self.accumulate_value / self.count


class MaxMetric(FnMetric):
	def __init__(self):
		super().__init__(lambda y_pred, y_true: y_pred.max(dim=1)[0])


class EqConfidenceMetric(Metrics):
	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, pred: Tensor, labels: Tensor):
		super().__call__(pred, labels)

		y_pred = (pred > self.confidence).float()
		y_true = (labels > self.confidence).float()

		self.value = (y_pred == y_true).all(dim=1).float().mean()
		self.accumulate_value += self.value
		return self.accumulate_value / self.count
