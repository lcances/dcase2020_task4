import torch

from dcase2020_task4.remixmatch.ModelDistributions import ModelDistributions


def test():
	nb_classes = 10
	distributions = ModelDistributions(nb_classes=nb_classes, max_samples=10)

	sample = torch.zeros(nb_classes)
	sample[1] = 1.0

	distributions.add_pred(sample, "labeled")

	for name in distributions.names:
		print("%s =" % name, distributions.get_mean_pred(name))


if __name__ == "__main__":
	test()
