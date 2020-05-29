import matplotlib.pyplot as plt
import numpy as np

from dcase2020_task4.util.rgb_augmentations import Inversion, UniColor, RandCrop, Gray


def get_demo_image():
	img = np.zeros((3, 128, 128))
	img[0] = np.linspace(start=list(range(128)), stop=list(range(128, 256)), num=128)
	img[1] = np.linspace(start=list(reversed(range(128))), stop=list(reversed(range(128, 256))), num=128)
	return img


def test():
	img = get_demo_image()

	augms = [Inversion(), UniColor(), RandCrop(), Gray()]

	plt.imshow(np.array(img.T, dtype=int))
	for augm in augms:
		img_a = augm(img.copy())

		plt.figure()
		plt.title(augm.__class__.__name__)
		plt.imshow(np.array(img_a.T, dtype=int))

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	test()
