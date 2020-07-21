import matplotlib.pyplot as plt
import numpy as np

from dcase2020_task4.util.other_img_augments import *

ratio = 1.0
ra_augments_set = [
	AutoContrast(ratio=ratio),
	Enhance(Brightness(ratio=ratio), levels=(0.05, 0.95)),
	Enhance(Color(ratio=ratio), levels=(0.05, 0.95)),
	Enhance(Contrast(ratio=ratio), levels=(0.05, 0.95)),
	Equalize(ratio=ratio),
	Identity(),
	Posterize(ratio=ratio, nbs_bits=(4, 8)),
	Rotation(ratio=ratio, rotation=(-np.pi / 6.0, np.pi / 6.0)),
	Enhance(Sharpness(ratio=ratio), levels=(0.05, 0.95)),
	ShearX(ratio=ratio, shears=(-0.3, 0.3)),
	ShearY(ratio=ratio, shears=(-0.3, 0.3)),
	Solarize(ratio=ratio, thresholds=(0, 255)),
	TranslateX(ratio=ratio, deltas=(-0.3, 0.3)),
	TranslateY(ratio=ratio, deltas=(-0.3, 0.3)),
]
cta_augments_set = [
	Blend(AutoContrast(ratio=ratio), levels=(0, 1)),
	Enhance(Brightness(ratio=ratio), levels=(0, 1)),
	Enhance(Color(ratio=ratio), levels=(0, 1)),
	Enhance(Contrast(ratio=ratio), levels=(0, 1)),
	Blend(Equalize(ratio=ratio), levels=(0, 1)),
	Blend(Invert(ratio=ratio), levels=(0, 1)),
	Identity(),
	Posterize(ratio=ratio, nbs_bits=(4, 8)),
	Rescale(ratio=ratio),
	Rotation(ratio=ratio, rotation=(-np.pi / 6.0, np.pi / 6.0)),
	Enhance(Sharpness(ratio=ratio), levels=(0, 1)),
	ShearX(ratio=ratio, shears=(-0.3, 0.3)),
	ShearY(ratio=ratio, shears=(-0.3, 0.3)),
	Blend(Smooth(ratio=ratio), levels=(0, 1)),
	Solarize(ratio=ratio, thresholds=(0, 255)),
	TranslateX(ratio=ratio, deltas=(-0.3, 0.3)),
	TranslateY(ratio=ratio, deltas=(-0.3, 0.3)),
]


def get_demo_image():
	img = np.zeros((3, 128, 128), dtype=np.uint8)
	img[0] = np.linspace(start=list(range(128)), stop=list(range(128, 256)), num=128)
	img[1] = np.linspace(start=list(reversed(range(128))), stop=list(reversed(range(128, 256))), num=128)
	img[2, 0:16] = 255
	return img


def test():
	img = get_demo_image()

	augms = [Inversion(), UniColor(), CutOut(), Gray()]

	plt.imshow(np.array(img.T, dtype=int))
	for augm in ra_augments_set:
		img_a = augm(img.copy().T).T

		print("DEBUG", img_a.shape)
		plt.figure()
		plt.title(augm.__class__.__name__)
		plt.imshow(np.array(img_a.T, dtype=int))

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	test()
