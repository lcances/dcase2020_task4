import torch

from dcase2020_task4.pytorch_metrics.metrics import Metrics


class MaxMetrics(Metrics):
	def __init__(self):
		super().__init__()

	def __call__(self, y_pred, y_true):
		super().__call__(y_pred, y_true)

		maxes = y_pred.max(dim=1)[0]
		self.value = torch.mean(maxes)
		self.accumulate_value += self.value

		return self.accumulate_value / self.count
