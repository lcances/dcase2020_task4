import numpy as np

from PIL import Image
from torchvision.transforms import Compose
from typing import Callable, List, Optional

from dcase2020_task4.util.other_img_augments import ImgRGBAugmentation, AutoContrast, Brightness, Color, Contrast, \
	Equalize, Posterize, Rotation, Sharpness, ShearX, ShearY, Solarize, TranslateX, TranslateY


RAND_AUGMENT_CLS_LIST = [
	AutoContrast,
	Brightness,
	Color,
	Contrast,
	Equalize,
	Posterize,
	Rotation,
	Sharpness,
	ShearX,
	ShearY,
	Solarize,
	TranslateX,
	TranslateY,
]


class RandAugment(ImgRGBAugmentation):
	def __init__(self, ratio: float = 1.0, magnitude_m: Optional[float] = 2.0, nb_choices_n: int = 1):
		"""
			@param ratio: probability of applying RandAugment.
			@param magnitude_m: Global magnitude constant. Use None for random magnitude.
			@param nb_choices_n: Nb augmentations applied.
		"""
		super().__init__(ratio)
		sub_ratio = 1.0

		self.magnitude_range = (0, 30)
		self.enhance_range = (0.05, 0.95)
		self.transforms_range = (-0.3, 0.3)
		self.posterize_range = (4, 8)
		self.angles_range = (-30, 30)
		self.thresholds_range = (0, 256)
		self.augments_list = [
			AutoContrast(ratio=sub_ratio),
			Brightness(ratio=sub_ratio, levels=self.enhance_range),
			Color(ratio=sub_ratio, levels=self.enhance_range),
			Contrast(ratio=sub_ratio, levels=self.enhance_range),
			Equalize(ratio=sub_ratio),
			Posterize(ratio=sub_ratio, nbs_bits=self.posterize_range),
			Rotation(ratio=sub_ratio, angles=self.angles_range),
			Sharpness(ratio=sub_ratio, levels=self.enhance_range),
			ShearX(ratio=sub_ratio, shears=self.transforms_range),
			ShearY(ratio=sub_ratio, shears=self.transforms_range),
			Solarize(ratio=sub_ratio, thresholds=self.thresholds_range),
			TranslateX(ratio=sub_ratio, deltas=self.transforms_range),
			TranslateY(ratio=sub_ratio, deltas=self.transforms_range),
		]

		self.magnitude = magnitude_m
		self.nb_choices_n = nb_choices_n

	def apply_helper(self, data: Image.Image) -> Image.Image:
		chosen = np.random.choice(self.augments_list, self.nb_choices_n)
		if self.magnitude is not None:
			chosen = self._apply_magnitude(chosen)
		return Compose(chosen)(data)

	def _apply_magnitude(self, chosen: List[Callable]) -> List[Callable]:
		for augm in chosen:
			if hasattr(augm, "enhance"):
				augm.enhance.levels = duplicate(to_range(self.magnitude, *self.enhance_range, *self.magnitude_range))
			elif hasattr(augm, "angles"):
				augm.angles = duplicate(to_range(self.magnitude, *self.angles_range, *self.magnitude_range))
			elif hasattr(augm, "nbs_bits"):
				augm.nbs_bits = duplicate(int(to_range(self.magnitude, *self.posterize_range, *self.magnitude_range)))
			elif hasattr(augm, "shears"):
				augm.shears = duplicate(to_range(self.magnitude, *self.transforms_range, *self.magnitude_range))
			elif hasattr(augm, "thresholds"):
				augm.thresholds = duplicate(int(to_range(self.magnitude, *self.thresholds_range, *self.magnitude_range)))
			elif hasattr(augm, "deltas"):
				augm.deltas = duplicate(to_range(self.magnitude, *self.transforms_range, *self.magnitude_range))
			elif isinstance(augm, AutoContrast) or isinstance(augm, Equalize):
				pass
			else:
				raise RuntimeError("Unknown augmentation \"%s\"." % augm.__name__)
		return chosen

	def _reset_ranges(self):
		for augm in self.augments_list:
			if hasattr(augm, "levels"):
				augm.levels = self.enhance_range
			elif hasattr(augm, "angles"):
				augm.angles = self.angles_range
			elif hasattr(augm, "nbs_bits"):
				augm.nbs_bits = self.posterize_range
			elif hasattr(augm, "shears"):
				augm.shears = self.transforms_range
			elif hasattr(augm, "thresholds"):
				augm.thresholds = self.thresholds_range
			elif hasattr(augm, "deltas"):
				augm.deltas = self.transforms_range
			elif isinstance(augm, AutoContrast) or isinstance(augm, Equalize):
				pass
			else:
				raise RuntimeError("Unknown augmentation.")


def to_range(value, new_min, new_max, old_min = 0, old_max = 1):
	return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def duplicate(value) -> tuple:
	return value, value
