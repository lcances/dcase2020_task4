import torch

from dcase2020_task4.util.avg_distributions import AvgDistributions


def test():
	nb_classes = 10
	distributions = AvgDistributions(
		history_size=10, shape=[nb_classes], names=["labeled", "unlabeled"], mode="multihot"
	)

	sample = torch.zeros(nb_classes)
	sample[1] = 1.0

	distributions.add_pred(sample, "labeled")

	for name in distributions.names:
		print("%s =" % name, distributions.get_avg_pred(name))

	pred = torch.ones(2, nb_classes).cuda() / nb_classes
	pred[:, 0] = 1.0

	print("Pred = ", pred)
	pred = distributions.apply_distribution_alignment(pred, dim=1)
	print("Pred = ", pred)


if __name__ == "__main__":
	test()
