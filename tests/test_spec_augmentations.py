import json
import matplotlib.pyplot as plt
import numpy as np

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip, Noise, RandomTimeDropout, RandomFreqDropout


def get_spec():
	with open("spec.json", "r") as file:
		data = json.load(file)
		x = data["x"]
		return np.array(x)


def test():
	spec = get_spec()

	ratio = 1.0
	augms = [
		# Transform(ratio, scale=(0.9, 1.1)),
		# Transform(ratio, rotation=(-np.pi / 2.0, np.pi / 2.0)),
		# Transform(ratio, translation=(-10, 10)),
		# TimeStretch(ratio),
		# HorizontalFlip(ratio),
		# VerticalFlip(ratio),
		Noise(ratio=ratio, snr=15.0),
		Noise2(ratio, noise_factor=(10.0, 10.0)),
		PitchShiftRandom(ratio),
		Occlusion(ratio, max_size=1.0),
		RandomFreqDropout(ratio, dropout=0.25),
		RandomTimeDropout(ratio, dropout=0.25),
	]

	plt.imshow(spec)
	for augm in augms:
		spec_a = augm(spec.copy())

		plt.figure()
		plt.title(augm.__class__.__name__)
		plt.imshow(spec_a)

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	test()
