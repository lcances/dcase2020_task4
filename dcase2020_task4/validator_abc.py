
from abc import ABC
from typing import Dict, List
from dcase2020_task4.metrics_recorder import MetricsRecorderABC
from metric_utils.metrics import Metrics


class ValidatorABC(ABC):
	def val(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def nb_examples(self) -> int:
		raise NotImplementedError("Abstract method")

	def get_all_metrics(self) -> List[Dict[str, Metrics]]:
		raise NotImplementedError("Abstract method")

	def reset_all_metrics(self):
		all_metrics = self.get_all_metrics()
		for metrics in all_metrics:
			for metric in metrics.values():
				metric.reset()

	def get_metrics_recorder(self) -> MetricsRecorderABC:
		raise NotImplementedError("Abstract method")
