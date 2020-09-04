"""
	WideResNet 28-2 class.
"""
import torch

from argparse import Namespace
from torch import Tensor, nn
from dcase2020_task4.other_models.wideresnet import ResNet, BasicBlock


class WideResNet28(ResNet):
	"""
		WideResNet-28 class.
	"""
	def __init__(self, num_classes: int, width: int = 2):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28':
		return WideResNet28(args.nb_classes)


class WideResNet28Rot(ResNet):
	"""
		WideResNet-28 class with rotation layer.
	"""
	def __init__(self, width: int = 2, rot_size: int = 4, num_classes: int = 10):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes)
		self.fc_rot = nn.Linear(64 * width * BasicBlock.expansion, rot_size)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28Rot':
		return WideResNet28Rot(rot_size=args.nb_classes_self_supervised, num_classes=args.nb_classes)

	def forward_rot(self, x: Tensor) -> Tensor:
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc_rot(x)

		return x
