import json
import matplotlib.pyplot as plt
import os.path as osp

from torchvision.datasets import CIFAR10
from dcase2020_task4.util.other_img_augments import *


ratio = 1.0
enhance_range = (1.0, 1.0)  # (0.05, 0.95)
# 14 augments
ra_augments_set = [
	AutoContrast(ratio=ratio),
	Brightness(ratio=ratio, levels=enhance_range),
	Color(ratio=ratio, levels=enhance_range),
	Contrast(ratio=ratio, levels=enhance_range),
	Equalize(ratio=ratio),
	Identity(),
	Posterize(ratio=ratio, nbs_bits=(4, 8)),
	Rotation(ratio=ratio, angles=(-30, 30)),
	Sharpness(ratio=ratio, levels=enhance_range),
	ShearX(ratio=ratio, shears=(-0.3, 0.3)),
	ShearY(ratio=ratio, shears=(-0.3, 0.3)),
	Solarize(ratio=ratio, thresholds=(0, 255)),
	TranslateX(ratio=ratio, deltas=(-0.3, 0.3)),
	TranslateY(ratio=ratio, deltas=(-0.3, 0.3)),
]
# 17 augments
cta_augments_set = [
	Blend(AutoContrast(ratio=ratio), levels=(0, 1)),
	Brightness(ratio=ratio, levels=(0, 1)),
	Color(ratio=ratio, levels=(0, 1)),
	Contrast(ratio=ratio, levels=(0, 1)),
	Blend(Equalize(ratio=ratio), levels=(0, 1)),
	Blend(Invert(ratio=ratio), levels=(0, 1)),
	Identity(),
	Posterize(ratio=ratio, nbs_bits=(1, 8)),
	Rescale(ratio=ratio),
	Rotation(ratio=ratio, angles=(-30, 30)),
	Sharpness(ratio=ratio, levels=(0, 1)),
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
	img[2, 0:16] = 128
	return img.T


def get_saved_img():
	filepath = "img.json"
	with open(filepath, "r") as file:
		data = json.load(file)
		x = data["x"]
		return np.array(x, dtype=np.uint8)


def save_cifar_img():
	dataset_path = osp.join("..", "dataset", "CIFAR10")
	dataset = CIFAR10(dataset_path, train=False, download=False, transform=np.array)

	rd = np.random.randint(0, len(dataset))
	img, label = dataset[rd]
	data = {"x": img.tolist(), "y": label, "index": rd}
	with open("img.json", "w") as file:
		json.dump(data, file, indent="\t")


def test():
	img = get_saved_img()

	augms = [
		Identity(),
		Rotation(ratio=ratio, angles=(-30, 30)),
		Inversion(),
		UniColor(),
		CutOut(),
		Gray()
	]
	print("Img original shape = %s (type=%s)" % (img.shape, img.dtype))

	for augm in augms:
		img_a = augm(img.copy())
		print("Img augm shape = %s (type=%s)" % (img_a.shape, img_a.dtype))

		plt.figure()
		plt.title(augm.__class__.__name__)
		plt.imshow(np.array(img_a, dtype=int))

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	test()
