import torch

from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy
from metric_utils.metrics import FScore


def test():
	metrics = [
		FScore(),
		BinaryConfidenceAccuracy(0.5),
	]

	pred = torch.zeros(5, 10)
	label = torch.zeros(5, 10)

	for metric in metrics:
		metric(pred, label)
		mean_ = metric(pred, label)
		print("Mean:", mean_)
		print("Value:", metric.value)


if __name__ == "__main__":
	test()
