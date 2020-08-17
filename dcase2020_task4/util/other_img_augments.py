import numpy as np
import torch

from abc import ABC
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torch import Tensor
from typing import List, Optional, Tuple

from augmentation_utils.augmentations import ImgAugmentation
from dcase2020_task4.util.utils_match import random_rect


class ImgRGBAugmentation(ABC, ImgAugmentation):
	""" Abstract class for images augmentations of size (width, height, 3). """

	def __init__(self, ratio: float, value_range: tuple = (0, 255)):
		super().__init__(ratio)
		self.value_range = value_range

	def _apply(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
		if len(data.shape) != 3 or data.shape[2] != 3:
			raise RuntimeError(
				"Invalid dimension %s. This augmentation only supports RGB (width, height, 3) images." %
				str(data.shape)
			)
		return self.apply_helper(data)


class Standardize(ImgRGBAugmentation):
	def __init__(self, means: List[float], stds: List[float], ratio: float = 1.0):
		super().__init__(ratio)
		self.means = list(means)
		self.stds = list(stds)

		if len(means) != len(stds):
			raise RuntimeError("Means and stds lists must have the same size.")

	def apply_helper(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
		if isinstance(data, Tensor):
			output = torch.zeros_like(data)
		else:
			output = np.zeros_like(data)

		for channel, (mean, std) in enumerate(zip(self.means, self.stds)):
			output[channel] = (data[channel] - mean) / std
		return output


class Gray(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: tuple = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
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

	def apply_helper(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
		width, height = data.shape[0], data.shape[1]
		r_left, r_right, r_top, r_down = random_rect(width, height, self.rect_width_scale_range, self.rect_height_scale_range)

		for i in range(data.shape[2]):
			data[r_left:r_right, r_top:r_down, i] = self.fill_value

		return data


class UniColor(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: tuple = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
		max_img = np.max(data, 2)

		color_chosen = np.random.randint(data.shape[2])
		for i in range(data.shape[2]):
			data[:, :, i] = max_img.copy() if i == color_chosen else self.value_range[0]

		return data


class Inversion(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, value_range: tuple = (0, 255)):
		super().__init__(ratio, value_range)

	def apply_helper(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
		return self.value_range[1] - data


# The following code of PIL augments is based on :
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py

class ImgPILAugmentation(ABC, ImgAugmentation):
	"""
		Abstract class that convert numpy array to PIL image for apply PIL augmentations internally.
		Image must have the size (width, height, 3).
	"""
	def __init__(self, ratio: float = 1.0, mode: str = "RGB"):
		ImgAugmentation.__init__(self, ratio)
		self.mode = mode

	def apply_helper(self, data: Image) -> Image.Image:
		raise NotImplementedError("Abstract method")

	def _apply(self, data: np.array) -> np.array:
		if len(data.shape) != 3 or data.shape[2] != 3:
			raise RuntimeError(
				"Invalid dimension %s. This augmentation only supports RGB (width, height, 3) images." %
				str(data.shape)
			)
		return np.asarray(self.apply_helper(Image.fromarray(data, mode=self.mode)))

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
		nb_bits = np.random.randint(*self.nbs_bits) if self.nbs_bits[0] != self.nbs_bits[1] else self.nbs_bits[0]
		return ImageOps.posterize(data, nb_bits)


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
		threshold = np.random.randint(*self.thresholds) if self.thresholds[0] != self.thresholds[1] else self.thresholds[0]
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
