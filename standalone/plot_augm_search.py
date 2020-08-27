import json
import numpy as np

from argparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt


def create_args() -> Namespace:
	parser = ArgumentParser()
	# results_augm_ubs8k, results_augm_cifar10
	parser.add_argument("--filepath", "-fp", type=str, default="../results/results_augm_ubs8k.json")
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

	kwargs = {full_name: ",".join([str(v) for v in params.values()]) for full_name, params in augments.items()}
	labels = [full_name.split("_")[0] + "\n(" + kwargs[full_name] + ")" for full_name in main_results.keys()]
	labels = [label for i, label in enumerate(labels)]

	augms_str = ["{} : {}".format(name, str(kwargs)) for name, kwargs in augments.items()]
	print("Augmentations : ", "\n".join(augms_str), "\n")
	print("Values : ", values)

	fig, ax = plt.subplots()
	rects = ax.barh(positions, values, label="Identity")

	ax.set_title("Validation accuracies")
	ax.set_xlabel("Categorical Accuracy")
	ax.set_yticks(positions)
	ax.set_yticklabels(labels)
	ax.legend()

	# Set values above bars
	for rect in rects:
		value = rect.get_width()
		ax.annotate(
			"{:.4f}".format(value),
			xy=(rect.get_width(), rect.get_y() + rect.get_height() / 4),
			xytext=(20, -2),  # horizontal & vertical offset
			textcoords="offset points",
			ha="center",
			va="bottom"
		)

	plt.show()


if __name__ == "__main__":
	main()
