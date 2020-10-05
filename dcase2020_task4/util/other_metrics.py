import torch

from torch import Tensor
from typing import Callable, Tuple, Union

from metric_utils.metrics import CategoricalAccuracy, Metrics


class CategoricalAccuracyOnehot(CategoricalAccuracy):
	""" Just Categorical Accuracy with a binarization with threshold. It takes one-hot vectors as input. """

	def __init__(self, dim: int, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.dim = dim

	def __call__(self, pred: Tensor, labels: Tensor) -> Tensor:
		with torch.no_grad():
			y_pred = pred.argmax(dim=self.dim)
			y_true = labels.argmax(dim=self.dim)
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
	def __init__(self, dim: int):
		super().__init__(lambda y_pred, y_true: y_pred.max(dim=dim)[0])


class MeanMetric(FnMetric):
	def __init__(self, dim: int):
		super().__init__(lambda y_pred, y_true: y_pred.mean(dim=dim))


class EqConfidenceMetric(Metrics):
	def __init__(self, confidence: float, dim: Union[int, Tuple[int, int]], epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence
		self.dim = dim

	def __call__(self, pred: Tensor, labels: Tensor) -> Tensor:
		super().__call__(pred, labels)

		with torch.no_grad():
			y_pred = (pred > self.confidence).float()
			y_true = (labels > self.confidence).float()

			self.value_ = (y_pred == y_true)
			if isinstance(self.dim, int):
				self.value_ = self.value_.all(dim=self.dim)
			else:
				for d in sorted(self.dim, reverse=True):
					self.value_ = self.value_.all(dim=d)
			self.value_ = self.value_.float().mean()
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
			y_true = (y_true > self.confidence).float()
			correct = (y_pred == y_true).float().sum()
			self.value_ = correct / torch.prod(torch.as_tensor(y_true.shape))

			self.accumulate_value += self.value_
			return self.accumulate_value / self.count
