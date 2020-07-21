import numpy as np

from abc import ABC
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Optional, Tuple

from augmentation_utils.augmentations import ImgAugmentation
from augmentation_utils.img_augmentations import Transform
from dcase2020_task4.util.utils_match import random_rect


class ImgRGBAugmentation(ABC, ImgAugmentation):
	""" Abstract class for images augmentations of size (3, width, height). """

	def __init__(self, ratio: float, value_range: list = (0, 255)):
		super().__init__(ratio)
		self.value_range = value_range

	def _apply(self, data):
		if len(data.shape) != 3 or data.shape[0] != 3:
			raise RuntimeError(
				"Invalid dimension %s. This augmentation only supports RGB (3, width, height) images." %
				str(data.shape)
			)
		return self.apply_helper(data)

	def apply_helper(self, data):
		raise NotImplementedError("Abstract method")


class Gray(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: list = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data):
		gray_img = np.mean(data, 0)
		for i in range(data.shape[0]):
			data[i] = gray_img.copy()
		return data


class CutOut(ImgRGBAugmentation):
	def __init__(
		self,
		ratio: float = 1.0,
		rect_min_scale: tuple = (0.1, 0.1),
		rect_max_scale: tuple = (0.5, 0.5),
		fill_value: Optional[int] = None,
		value_range: list = (0, 255),
	):
		super().__init__(ratio, value_range)
		self.rect_min_scale = rect_min_scale
		self.rect_max_scale = rect_max_scale
		self.fill_value = fill_value

	def apply_helper(self, data):
		width, height = data.shape[1], data.shape[2]
		r_left, r_right, r_top, r_down = random_rect(width, height, self.rect_min_scale, self.rect_max_scale)

		fill_value = self.get_fill_value()
		for i in range(data.shape[0]):
			data[i, r_left:r_right, r_top:r_down] = fill_value

		return data

	def get_fill_value(self) -> int:
		if self.fill_value is None:
			return int((self.value_range[1] + self.value_range[0]) / 2.0)
		else:
			return self.fill_value


class UniColor(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: list = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data):
		max_img = np.max(data, 0)

		color_chosen = np.random.randint(data.shape[0])
		for i in range(data.shape[0]):
			if i != color_chosen:
				data[i] = self.value_range[0]
			else:
				data[i] = max_img.copy()

		return data


class Inversion(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: list = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data):
		return self.value_range[1] - data


def autocontrast(x, level):
	return _imageop(x, ImageOps.autocontrast, level)


def blur(x, level):
	return _filter(x, ImageFilter.BLUR, level)


def brightness(x, brightness):
	return _enhance(x, ImageEnhance.Brightness, brightness)


def color(x, color):
	return _enhance(x, ImageEnhance.Color, color)


def contrast(x, contrast):
	return _enhance(x, ImageEnhance.Contrast, contrast)


def cutout(x, level):
	"""Apply cutout to pil_img at the specified level."""
	size = 1 + int(level * min(x.size) * 0.499)
	img_height, img_width = x.size
	height_loc = np.random.randint(low=0, high=img_height)
	width_loc = np.random.randint(low=0, high=img_width)
	upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
	lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
	pixels = x.load()  # create the pixel map
	for i in range(upper_coord[0], lower_coord[0]):  # for every col:
		for j in range(upper_coord[1], lower_coord[1]):  # For every row
			pixels[i, j] = (127, 127, 127)  # set the color accordingly
	return x


def equalize(x, level):
	return _imageop(x, ImageOps.equalize, level)


def invert(x, level):
	return _imageop(x, ImageOps.invert, level)


def identity(x):
	return x


def posterize(x, level):
	level = 1 + int(level * 7.999)
	return ImageOps.posterize(x, level)


def rescale(x, scale, method):
	s = x.size
	scale *= 0.25
	crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
	methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
	method = methods[int(method * 5.99)]
	return x.crop(crop).resize(x.size, method)


def rotate(x, angle):
	angle = int(np.round((2 * angle - 1) * 45))
	return x.rotate(angle)


def sharpness(x, sharpness):
	return _enhance(x, ImageEnhance.Sharpness, sharpness)


def shear_x(x, shear):
	shear = (2 * shear - 1) * 0.3
	return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


def shear_y(x, shear):
	shear = (2 * shear - 1) * 0.3
	return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


def smooth(x, level):
	return _filter(x, ImageFilter.SMOOTH, level)


def solarize(x, th):
	th = int(th * 255.999)
	return ImageOps.solarize(x, th)


def translate_x(x, delta):
	delta = (2 * delta - 1) * 0.3
	return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


def translate_y(x, delta):
	delta = (2 * delta - 1) * 0.3
	return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


class ImgPILAugmentation(ABC, ImgAugmentation):
	def _apply(self, data: np.array):
		if len(data.shape) != 3 or data.shape[2] != 3:
			raise RuntimeError("Unsupported image shape.")
		return np.array(self.apply_helper(Image.fromarray(data, mode="RGB")))


class Blend(ImgPILAugmentation):
	def __init__(self, augment: ImgPILAugmentation, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.augment = augment
		self.levels = levels

	def apply_helper(self, data: Image.Image) -> Image.Image:
		level = np.random.uniform(*self.levels)
		return Image.blend(data, self.augment(np.asarray(data)), level)


class Enhance(ImgPILAugmentation):
	def __init__(self, augment: ImgPILAugmentation, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.augment = augment
		self.levels = levels

	def apply_helper(self, data):
		level = np.random.uniform(*self.levels)
		return self.augment.apply_helper(data).enhance(0.1 + 1.9 * level)


class AutoContrast(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageOps.autocontrast(data)


class Blur(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return np.array(Image.fromarray(data).filter(ImageFilter.BLUR))


class Brightness(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageEnhance.Brightness(data)


class Color(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageEnhance.Color(data)


class Contrast(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageEnhance.Contrast(data)


class Equalize(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageOps.equalize(data)


class Invert(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageOps.invert(data)


class Identity(ImgPILAugmentation):
	def __init__(self):
		super().__init__(1.0)

	def apply_helper(self, data):
		return data


class Posterize(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, nbs_bits: Tuple[int, int] = (1, 8)):
		super().__init__(ratio)
		self.nbs_bits = nbs_bits

	def apply_helper(self, data):
		nb_bits = np.random.randint(*self.nbs_bits)
		return ImageOps.posterize(data, nb_bits)


class Rescale(Transform):
	def __init__(self, ratio: float = 1.0, scale: Tuple[float, float] = (1.0, 1.0)):
		super().__init__(ratio=ratio, scale=scale)


class Rotation(Transform):
	def __init__(self, ratio: float = 1.0, rotation: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio=ratio, rotation=rotation)


class Sharpness(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return ImageEnhance.Sharpness(data)


class ShearX(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, shears: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.shears = shears

	def apply_helper(self, data):
		shear = np.random.uniform(*self.shears)
		return data.transform(data.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


class ShearY(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, shears: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.shears = shears

	def apply_helper(self, data):
		shear = np.random.uniform(*self.shears)
		return data.transform(data.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


class Smooth(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return data.filter(ImageFilter.SMOOTH)


class Solarize(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, thresholds: Tuple[int, int] = (0, 255)):
		super().__init__(ratio)
		self.thresholds = thresholds

	def apply_helper(self, data):
		threshold = np.random.randint(*self.thresholds)
		return ImageOps.solarize(data, threshold)


class TranslateX(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, deltas: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.deltas = deltas

	def apply_helper(self, data):
		delta = np.random.uniform(*self.deltas)
		delta *= data.size[1]
		return data.transform(data.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


class TranslateY(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, deltas: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.deltas = deltas

	def apply_helper(self, data):
		delta = np.random.uniform(*self.deltas)
		delta *= data.size[1]
		return data.transform(data.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


def _enhance(x, op, level):
	return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
	return Image.blend(x, op(x), level)


def _filter(x, op, level):
	return Image.blend(x, x.filter(op), level)


def pil_to_np(img: Image) -> np.array:
	return np.asarray(img)


def np_to_pil(img: np.array) -> Image:
	return Image.fromarray(img)
