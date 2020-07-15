import json
import numpy as np
from matplotlib import pyplot as plt


def main():
	filepath = "../osirim_labbeti/results_augm.json"
	with open(filepath, "r") as file:
		data = json.load(file)

	results = data["results"]
	augments = data["augments"]
	augm_idx = {k: i for i, k in enumerate(results.keys())}

	results_mat = np.zeros((len(augm_idx), len(augm_idx)), dtype=float)
	for k1, i1 in augm_idx.items():
		for k2, i2 in augm_idx.items():
			results_mat[i1][i2] = results[k1][k2]

	augms_str = ["{} : {}".format(name, str(kwargs)) for name, kwargs in augments.items()]
	print("Augmentations : ", "\n".join(augms_str))

	values = results_mat[0]
	labels = [k.split("_")[0] for k in results.keys()]

	positions = np.arange(len(labels))

	fig, ax = plt.subplots()
	rects = ax.bar(positions, values, label="Identity")

	ax.set_ylabel("Categorical Accuracy")
	ax.set_title("Validation accuracies")
	ax.set_xticks(positions)
	ax.set_xticklabels(labels)
	ax.legend()

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
