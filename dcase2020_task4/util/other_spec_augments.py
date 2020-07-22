
from typing import Optional

from augmentation_utils.augmentations import SpecAugmentation
from dcase2020_task4.util.utils_match import random_rect


class CutOutSpec(SpecAugmentation):
	def __init__(
		self,
		ratio: float = 1.0,
		rect_width_scale_range: tuple = (0.1, 0.5),
		rect_height_scale_range: tuple = (0.1, 0.5),
		fill_value: Optional[int] = None
	):
		super().__init__(ratio)
		self.value_range = (-80.0, 0.0)
		self.rect_width_scale_range = rect_width_scale_range
		self.rect_height_scale_range = rect_height_scale_range
		self.fill_value = fill_value if fill_value is not None else int((self.value_range[1] + self.value_range[0]) / 2.0)

	def apply_helper(self, data):
		width, height = data.shape[0], data.shape[1]
		r_left, r_right, r_top, r_down = random_rect(width, height, self.rect_width_scale_range, self.rect_height_scale_range)

		data[r_left:r_right, r_top:r_down] = self.fill_value

		return data


class InversionSpec(SpecAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)
		self.value_range = (-80.0, 0.0)

	def apply_helper(self, data):
		return self.value_range[1] + self.value_range[0] - data
