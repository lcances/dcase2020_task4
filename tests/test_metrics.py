import torch

from dcase2020_task4.util.other_metrics import BinaryConfidenceAccuracy, BestMetric, MaxMetric
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


def test_2():
	metric = FScore()
	bmetric = BestMetric(metric)

	label = torch.ones(5)
	for e in range(5):
		pred = label * (e / 5)
		mean_ = metric(pred, label)
		bmean = bmetric(pred, label)

		print(mean_)
		print(bmean)


def test_3():
	metric = MaxMetric()
	pred = torch.zeros(5, 3)
	for i in range(5):
		pred[i, 1] = i - 1
		pred[i, 0] = i
	label = torch.zeros(5, 10)

	mean_ = metric(pred, label)
	print(metric.value)
	print(mean_)

	for i in range(5):
		pred[i, 0] = 5 - i
		pred[i, 1] = 5 - i - 1

	mean_ = metric(pred, label)
	print(metric.value)
	print(mean_)

	metric.reset()

	mean_ = metric(pred, label)
	print(metric.value)
	print(mean_)


if __name__ == "__main__":
	test_3()
