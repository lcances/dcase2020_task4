"""
	WideResNet 28-2 class.
"""

from argparse import Namespace
from dcase2020_task4.other_models.wideresnet import ResNet


class WideResNet28_2(ResNet):
	"""
		WideResNet-28-2 class.
	"""
	def __init__(self, num_classes: int):
		super().__init__(layers=[4, 4, 4], width=2, num_classes=num_classes)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28_2':
		return WideResNet28_2(args.nb_classes)
