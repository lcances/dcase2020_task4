import numpy as np

from abc import ABC
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision.transforms import RandomChoice
from typing import Optional, Tuple

from augmentation_utils.augmentations import ImgAugmentation
from augmentation_utils.img_augmentations import Transform
from dcase2020_task4.util.utils_match import random_rect


class ImgRGBAugmentation(ABC, ImgAugmentation):
	""" Abstract class for images augmentations of size (width, height, 3). """

	def __init__(self, ratio: float, value_range: tuple = (0, 255)):
		super().__init__(ratio)
		self.value_range = value_range

	def _apply(self, data: np.array) -> np.array:
		if len(data.shape) != 3 or data.shape[2] != 3:
			raise RuntimeError(
				"Invalid dimension %s. This augmentation only supports RGB (width, height, 3) images." %
				str(data.shape)
			)
		return self.apply_helper(data)


class Gray(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: tuple = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data):
		# Mean on dimension 2
		gray_img = data.mean(2)
		for i in range(data.shape[2]):
			data[:, :, i] = gray_img.copy()
		return data


class CutOut(ImgRGBAugmentation):
	def __init__(
		self,
		ratio: float = 1.0,
		rect_width_scale_range: tuple = (0.1, 0.5),
		rect_height_scale_range: tuple = (0.1, 0.5),
		fill_value: Optional[int] = None,
		value_range: tuple = (0, 255),
	):
		super().__init__(ratio, value_range)
		self.rect_width_scale_range = rect_width_scale_range
		self.rect_height_scale_range = rect_height_scale_range
		self.fill_value = fill_value if fill_value is not None else int((self.value_range[1] + self.value_range[0]) / 2.0)

	def apply_helper(self, data):
		width, height = data.shape[0], data.shape[1]
		r_left, r_right, r_top, r_down = random_rect(width, height, self.rect_width_scale_range, self.rect_height_scale_range)

		for i in range(data.shape[2]):
			data[r_left:r_right, r_top:r_down, i] = self.fill_value

		return data


class UniColor(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: tuple = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data):
		max_img = np.max(data, 2)

		color_chosen = np.random.randint(data.shape[2])
		for i in range(data.shape[2]):
			data[:, :, i] = max_img.copy() if i == color_chosen else self.value_range[0]

		return data


class Inversion(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: tuple = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data):
		return self.value_range[1] - data


class ImgPILAugmentation(ABC, ImgAugmentation):
	"""
		Abstract class that convert numpy array to PIL image for apply PIL augmentations.
		Image must have the size (width, height, 3)
	"""
	def __init__(self, ratio: float = 1.0, mode: str = "RGB"):
		ImgAugmentation.__init__(self, ratio)
		self.mode = mode

	def _apply(self, data: np.array) -> np.array:
		if len(data.shape) != 3 or data.shape[2] != 3:
			raise RuntimeError(
				"Invalid dimension %s. This augmentation only supports RGB (width, height, 3) images." %
				str(data.shape)
			)
		return np.asarray(self.apply_helper(Image.fromarray(data, mode=self.mode)))

	def apply_helper(self, data: Image.Image) -> Image.Image:
		raise NotImplementedError("Abstract method")

	def set_mode(self, mode: str):
		self.mode = mode

	def get_mode(self) -> str:
		return self.mode


class Enhance(ImgPILAugmentation):
	def __init__(self, method: "ImageEnhance", ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.method = method
		self.levels = levels

	def apply_helper(self, data: Image.Image) -> Image.Image:
		level = np.random.uniform(*self.levels)
		return self.method(data).enhance(0.1 + 1.9 * level)


class Blend(ImgPILAugmentation):
	def __init__(self, augment: ImgPILAugmentation, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.augment = augment
		self.levels = levels

	def apply_helper(self, data: Image.Image) -> Image.Image:
		level = np.random.uniform(*self.levels)
		return Image.blend(data, self.augment.apply_helper(data), level)


class AutoContrast(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return ImageOps.autocontrast(data)


class Blur(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return data.filter(ImageFilter.BLUR)


class Brightness(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.enhance = Enhance(method=ImageEnhance.Brightness, ratio=1.0, levels=levels)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply_helper(data)


class Color(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.enhance = Enhance(method=ImageEnhance.Color, ratio=1.0, levels=levels)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply_helper(data)


class Contrast(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.enhance = Enhance(method=ImageEnhance.Contrast, ratio=1.0, levels=levels)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply_helper(data)


class Equalize(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return ImageOps.equalize(data)


class Invert(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return ImageOps.invert(data)


class Identity(ImgPILAugmentation):
	def __init__(self):
		super().__init__(1.0)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return data


class Posterize(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, nbs_bits: Tuple[int, int] = (1, 8)):
		super().__init__(ratio)
		self.nbs_bits = nbs_bits

	def apply_helper(self, data: Image.Image) -> Image.Image:
		nb_bits = np.random.randint(*self.nbs_bits)
		return ImageOps.posterize(data, nb_bits)


class RescaleOld(Transform):
	def __init__(self, ratio: float = 1.0, scales: Tuple[float, float] = (1.0, 1.0)):
		super().__init__(ratio=ratio, scale=scales)


class Rescale(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, scales: Tuple[float, float] = (1.0, 1.0), method: int = Image.NEAREST):
		"""
			Available methods :
				Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST
		"""
		super().__init__(ratio=ratio)
		self.scales = scales
		self.method = method

	def apply_helper(self, data: Image.Image) -> Image.Image:
		scale = np.random.uniform(*self.scales)
		scale -= 1
		scale *= (1.0 if scale <= 0.0 else 0.25)
		size = data.size
		crop = (scale * size[0], scale * size[1], size[0] * (1 - scale), size[1] * (1 - scale))
		return data.crop(crop).resize(data.size, self.method)


class Rotation(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, angles: Tuple[float, float] = (0.0, 0.0)):
		"""
			Rotate an image using PIL methods.
			Angles must be in degrees in range [-180, 180].
			Uniformly sample angle of rotation from range "angles".
		"""
		super().__init__(ratio=ratio)
		self.angles = angles

	def apply_helper(self, data: Image.Image) -> Image.Image:
		angle = np.random.uniform(*self.angles)
		return data.rotate(angle)


class Sharpness(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, levels: Tuple[float, float] = (0.5, 0.5)):
		super().__init__(ratio)
		self.enhance = Enhance(method=ImageEnhance.Sharpness, ratio=1.0, levels=levels)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return self.enhance.apply_helper(data)


class ShearX(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, shears: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.shears = shears

	def apply_helper(self, data: Image.Image) -> Image.Image:
		shear = np.random.uniform(*self.shears)
		return data.transform(data.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


class ShearY(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, shears: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.shears = shears

	def apply_helper(self, data: Image.Image) -> Image.Image:
		shear = np.random.uniform(*self.shears)
		return data.transform(data.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


class Smooth(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data: Image.Image) -> Image.Image:
		return data.filter(ImageFilter.SMOOTH)


class Solarize(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, thresholds: Tuple[int, int] = (0, 255)):
		super().__init__(ratio)
		self.thresholds = thresholds

	def apply_helper(self, data: Image.Image) -> Image.Image:
		threshold = np.random.randint(*self.thresholds)
		return ImageOps.solarize(data, threshold)


class TranslateX(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, deltas: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.deltas = deltas

	def apply_helper(self, data: Image.Image) -> Image.Image:
		delta = np.random.uniform(*self.deltas)
		delta *= data.size[1]
		return data.transform(data.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


class TranslateY(ImgPILAugmentation):
	def __init__(self, ratio: float = 1.0, deltas: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(ratio)
		self.deltas = deltas

	def apply_helper(self, data: Image.Image) -> Image.Image:
		delta = np.random.uniform(*self.deltas)
		delta *= data.size[1]
		return data.transform(data.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


# Note: AUGMENTS IMPORT FROM
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py


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


def _enhance(x, op, level):
	return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
	return Image.blend(x, op(x), level)


def _filter(x, op, level):
	return Image.blend(x, x.filter(op), level)


class RandAugment(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, magnitude: float = 0.5):
		super().__init__(ratio)
		sub_ratio = 1.0

		enhance_range = (0.05, 0.95)
		transforms_range = (-0.3, 0.3)
		posterize_range = (4, 8)
		angles_range = (-30, 30)
		thresholds_range = (0, 256)

		self.magnitude = magnitude
		self.augment_fn = RandomChoice([
			AutoContrast(ratio=sub_ratio),
			Brightness(ratio=sub_ratio, levels=enhance_range),
			Color(ratio=sub_ratio, levels=enhance_range),
			Contrast(ratio=sub_ratio, levels=enhance_range),
			Equalize(ratio=sub_ratio),
			Posterize(ratio=sub_ratio, nbs_bits=posterize_range),
			Rotation(ratio=sub_ratio, angles=angles_range),
			Sharpness(ratio=sub_ratio, levels=enhance_range),
			ShearX(ratio=sub_ratio, shears=transforms_range),
			ShearY(ratio=sub_ratio, shears=transforms_range),
			Solarize(ratio=sub_ratio, thresholds=thresholds_range),
			TranslateX(ratio=sub_ratio, deltas=transforms_range),
			TranslateY(ratio=sub_ratio, deltas=transforms_range),
		])

	def apply_helper(self, data):
		return self.augment_fn(data)


def to_range(value, min_, max_):
	return value * (max_ - min_) + min_


def duplicate(value) -> tuple:
	return value, value
