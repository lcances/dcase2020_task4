
from typing import Optional

from augmentation_utils.augmentations import SpecAugmentation
from dcase2020_task4.util.utils_match import random_rect


class CutOutSpec(SpecAugmentation):
	def __init__(
		self,
		ratio: float = 1.0,
		rect_min_scale: tuple = (0.1, 0.1),
		rect_max_scale: tuple = (0.5, 0.5),
		fill_value: Optional[int] = None
	):
		super().__init__(ratio)
		self.rect_min_scale = rect_min_scale
		self.rect_max_scale = rect_max_scale
		self.fill_value = fill_value
		self.value_range = (-80.0, 0.0)

	def apply_helper(self, data):
		width, height = data.shape[0], data.shape[1]
		r_left, r_right, r_top, r_down = random_rect(width, height, self.rect_min_scale, self.rect_max_scale)

		fill_value = self.get_fill_value()
		data[r_left:r_right, r_top:r_down] = fill_value

		return data

	def get_fill_value(self) -> int:
		if self.fill_value is None:
			return int((self.value_range[1] + self.value_range[0]) / 2.0)
		else:
			return self.fill_value


class InversionSpec(SpecAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)
		self.value_range = (-80.0, 0.0)

	def apply_helper(self, data):
		return self.value_range[1] + self.value_range[0] - data
