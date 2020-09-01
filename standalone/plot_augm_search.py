import json
import numpy as np
import os.path as osp

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

	main_results = dict(sorted(main_results.items(), key=lambda item: item[1]))
	values = list(main_results.values())

	kwargs = {full_name: ",".join([str(v) for v in params.values()]) for full_name, params in augments.items()}
	labels = [full_name.split("_")[0] + "\n(" + kwargs[full_name] + ")" for full_name in main_results.keys()]
	labels = [label for i, label in enumerate(labels)]

	augms_str = ["{} : {}".format(name, str(kwargs)) for name, kwargs in augments.items()]
	print("Augmentations : ", "\n".join(augms_str), "\n")
	print("Values : ", values)

	fig, ax = plt.subplots()
	positions = np.arange(len(values))
	rects = ax.bar(positions, values)  # , label="Identity"

	plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

	ax.set_title("Validation accuracies")
	ax.set_xticks(positions)
	ax.set_xticklabels(labels)
	ax.set_ylabel("Categorical Accuracy")
	ax.legend()

	# Set values above bars
	for rect in rects:
		value = rect.get_height()
		ax.annotate(
			"{:.4f}".format(value),
			xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
			xytext=(0, 0),  # horizontal & vertical offset
			textcoords="offset points",
			ha="center",
			va="bottom"
		)

	manager = plt.get_current_fig_manager()
	manager.window.showMaximized()
	plt.show()

	dirpath = osp.join("..", "results", "img")
	prefix = "augm_search"
	name = "spec"
	filepath = osp.join(dirpath, "%s_%s.png" % (prefix, name))
	fig.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)

	plt.show()


if __name__ == "__main__":
	main()
