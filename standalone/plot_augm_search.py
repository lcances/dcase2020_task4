import json
import numpy as np

from argparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt


def create_args() -> Namespace:
	parser = ArgumentParser()
	# results_augm_ubs8k, results_augm_cifar10
	parser.add_argument("--filepath", "-fp", type=str, default="../labbeti_osirim/results_augm_ubs8k.json")
	return parser.parse_args()


def main():
	args = create_args()
	with open(args.filepath, "r") as file:
		data = json.load(file)

	results = data["results"]
	augments = data["augments"]

	main_augm = "Identity_"
	main_results = results[main_augm]

	positions = np.arange(len(main_results))
	values = list(main_results.values())
	labels = [k.split("_")[0] for k in main_results.keys()]

	labels = [("\n" * (i % 3)) + label for i, label in enumerate(labels)]

	augms_str = ["{} : {}".format(name, str(kwargs)) for name, kwargs in augments.items()]
	print("Augmentations : ", "\n".join(augms_str), "\n")
	print("Values : ", values)

	fig, ax = plt.subplots()
	rects = ax.bar(positions, values, label="Identity")

	ax.set_ylabel("Categorical Accuracy")
	ax.set_title("Validation accuracies")
	ax.set_xticks(positions)
	ax.set_xticklabels(labels)
	ax.legend()

	# Set values above bars
	for rect in rects:
		bar_height = rect.get_height()
		ax.annotate('{:.2f}'.format(bar_height),
					xy=(rect.get_x() + rect.get_width() / 2, bar_height),
					xytext=(0, 3),  # 3 points vertical offset
					textcoords="offset points",
					ha='center', va='bottom')

	plt.show()


if __name__ == "__main__":
	main()
