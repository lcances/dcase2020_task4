import torch

from dcase2020_task4.util.avg_distributions import AvgDistributions


def test():
	nb_classes = 10
	distributions = AvgDistributions(
		history_size=10, nb_classes=nb_classes, names=["labeled"], mode="onehot"
	)

	sample = torch.zeros(nb_classes)
	sample[1] = 1.0

	distributions.add_pred(sample, "labeled")

	for name in distributions.names:
		print("%s =" % name, distributions.get_avg_pred(name))


if __name__ == "__main__":
	test()
