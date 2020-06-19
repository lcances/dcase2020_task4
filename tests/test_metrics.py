import torch

from metric_utils.metrics import FScore


def test():
	metrics = [
		FScore(),
	]

	pred = torch.zeros(5, 10)
	label = torch.zeros(5, 10)
	for metric in metrics:
		metric(pred, label)
		print(metric.value_.item())


if __name__ == "__main__":
	test()
