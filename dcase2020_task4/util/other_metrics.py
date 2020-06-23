import torch

from torch import Tensor
from typing import Callable
from metric_utils.metrics import CategoricalAccuracy, Metrics


class CategoricalConfidenceAccuracy(CategoricalAccuracy):
	""" Just Categorical Accuracy with a binarization with threshold. """

	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, pred: Tensor, labels: Tensor) -> Tensor:
		with torch.no_grad():
			y_pred = (pred > self.confidence).float()
			y_true = (labels > self.confidence).float()
			return super().__call__(y_pred, y_true)


class FnMetric(Metrics):
	def __init__(self, fn: Callable[[Tensor, Tensor], Tensor]):
		super().__init__()
		self.fn = fn

	def __call__(self, pred: Tensor, labels: Tensor) -> Tensor:
		super().__call__(pred, labels)

		with torch.no_grad():
			self.value_ = self.fn(pred, labels).mean()
			self.accumulate_value += self.value_

			return self.accumulate_value / self.count


class MaxMetric(FnMetric):
	def __init__(self):
		super().__init__(lambda y_pred, y_true: y_pred.max(dim=1)[0])


class MeanMetric(FnMetric):
	def __init__(self):
		super().__init__(lambda y_pred, y_true: y_pred.mean(dim=1))


class EqConfidenceMetric(Metrics):
	def __init__(self, confidence: float, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence

	def __call__(self, pred: Tensor, labels: Tensor) -> Tensor:
		super().__call__(pred, labels)

		with torch.no_grad():
			y_pred = (pred > self.confidence).float()
			y_true = (labels > self.confidence).float()

			self.value_ = (y_pred == y_true).all(dim=1).float().mean()
			self.accumulate_value += self.value_
			return self.accumulate_value / self.count


class BinaryConfidenceAccuracy(Metrics):
	def __init__(self, confidence: float, epsilon: float = 1e-10):
		Metrics.__init__(self, epsilon)
		self.confidence = confidence

	def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
		super().__call__(y_pred, y_true)

		with torch.no_grad():
			y_pred = (y_pred > self.confidence).float()
			correct = (y_pred == y_true).float().sum()
			self.value_ = correct / torch.prod(torch.as_tensor(y_true.shape))

			self.accumulate_value += self.value_
			return self.accumulate_value / self.count


class BestMetric(Metrics):
	# TODO : fix
	def __init__(self, metric: Metrics, sub_call: bool = False, mode: str = "max"):
		super().__init__()
		self.best = torch.as_tensor(0) if mode == "max" else torch.as_tensor(2**20)
		self.metric = metric
		self.sub_call = sub_call
		self.mode = mode

	def __call__(self, pred: Tensor, label: Tensor) -> Tensor:
		if self.sub_call:
			self.metric(pred, label)
		self.accumulate_value = self.metric.accumulate_value
		self.count = self.metric.count
		return self.best

	@property
	def value(self):
		return self.best

	def reset(self):
		super().reset()
		mean_ = self.accumulate_value / self.count if self.count != 0 else 0
		if self.mode == "max" and mean_ > self.best:
			self.best = mean_
		elif self.mode == "min" and mean_ < self.best:
			self.best = mean_
