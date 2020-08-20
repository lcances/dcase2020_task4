import json
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import RandomChoice, Compose
from augmentation_utils.img_augmentations import Transform
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip, Noise, RandomTimeDropout, RandomFreqDropout
from dcase2020_task4.util.other_spec_augments import IdentitySpec, CutOutSpec, InversionSpec


def get_spec():
	with open("spec.json", "r") as file:
		data = json.load(file)
		x = data["x"]
		return np.array(x)


def test():
	spec = get_spec()

	ratio = 1.0
	augms = [
		IdentitySpec(),
		# Transform(ratio, scale=(1.1, 1.1)),
		# Transform(ratio, rotation=(0.2, 0.2)),
		# Transform(ratio, rotation=(0.6, 0.6)),
		# Transform(ratio, rotation=(-np.pi / 4.0, np.pi / 4.0)),
		# Transform(ratio, translation=(-10, 10)),
		# TimeStretch(ratio),
		# PitchShiftRandom(ratio),
		# PitchShiftRandom(ratio, steps=(-1, 1)),
		# Noise(ratio=ratio, snr=15.0),
		# Noise(ratio=ratio, snr=5.0),
		# Noise2(ratio, noise_factor=(10.0, 10.0)),
		# Noise2(ratio, noise_factor=(5.0, 5.0)),
		# Occlusion(ratio, max_size=1.0),
		# RandomFreqDropout(ratio, dropout=0.01),
		# RandomTimeDropout(ratio, dropout=0.25),
		# RandCropSpec(ratio, fill_value=-80),
		# VerticalFlip(ratio),
		# Compose([HorizontalFlip(ratio), VerticalFlip(ratio)]),
		HorizontalFlip(ratio),
		# CutOutSpec(ratio),
		# InversionSpec(ratio),
	]
	print(spec.shape)

	for augm in augms:
		spec_a = augm(spec.copy())

		plt.figure()
		plt.title(augm.__class__.__name__)
		plt.imshow(spec_a)

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	test()
