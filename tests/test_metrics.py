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

	shape = 5
	label = torch.rand(shape)
	for e in range(5):
		metric.reset()
		bmetric.reset()

		for i in range(3):
			pred = label * (e / 5) + torch.rand(shape) * 10

			mean_ = metric(pred, label)
			bmean = bmetric(pred, label)

			print("e %d, i %d" % (e, i))
			print("Mean: ", mean_.item())
			print("Value: ", metric.value.item())
			print("BMean: ", bmean.item())
			print("BValue: ", bmetric.value.item())
			print("")


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
	test_2()
