"""
	CODE IMPORTED FROM https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from torch import Tensor


relu_fn = nn.ReLU  # nn.ReLU, nn.LeakyReLU
bn_momentum = 0.1  # 0.1, 0.999


class BasicBlock(nn.Module):
	def __init__(self, in_planes, out_planes, stride, dropout: float = 0.0):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
		self.relu1 = relu_fn(inplace=True)
		self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
		self.relu2 = relu_fn(inplace=True)
		self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
							   padding=1, bias=False)
		self.dropout = dropout
		self.equalInOut = (in_planes == out_planes)
		self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
							   padding=0, bias=False) or None

	def forward(self, x):
		if not self.equalInOut:
			x = self.relu1(self.bn1(x))
			out = x
		else:
			out = self.relu1(self.bn1(x))
		out = self.relu2(self.bn2(self.conv1(out)))
		if self.dropout > 0:
			out = F.dropout(out, p=self.dropout, training=self.training)
		out = self.conv2(out)
		return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout: float = 0.0):
		super(NetworkBlock, self).__init__()
		self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout)

	def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout: float):
		layers = []
		for i in range(int(nb_layers)):
			layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout))
		return nn.Sequential(*layers)

	def forward(self, x):
		return self.layer(x)


class WideResNet(nn.Module):
	def __init__(self, depth: int = 28, num_classes: int = 10, widen_factor: int = 2, dropout: float = 0.5):
		# TODO : old widen_factor = 1
		super(WideResNet, self).__init__()
		n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
		assert((depth - 4) % 6 == 0)
		n = (depth - 4) / 6
		block = BasicBlock
		# 1st conv before any network block
		self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
		# 1st block
		self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, dropout)
		# 2nd block
		self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropout)
		# 3rd block
		self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropout)
		# global average pooling and classifier
		self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=bn_momentum)
		self.relu = relu_fn(inplace=True)
		self.fc = nn.Linear(n_channels[3], num_classes)
		self.nChannels = n_channels[3]

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet':
		return WideResNet(
			depth=args.wrn_depth,
			num_classes=args.nb_classes,
			widen_factor=args.wrn_widen_factor,
			dropout=args.dropout,
		)

	def forward(self, x: Tensor) -> Tensor:
		out = self._features(x)
		out = self.fc(out)
		return out

	def _features(self, x: Tensor) -> Tensor:
		out = self.conv1(x)
		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.relu(self.bn1(out))
		out = F.avg_pool2d(out, 8)
		out = out.view(-1, self.nChannels)
		return out


class WideResNetRot(WideResNet):
	def __init__(
		self, depth: int, num_classes: int = 10, widen_factor: int = 2, dropout: float = 0.5, rot_output_size: int = 4
	):
		super().__init__(depth, num_classes, widen_factor, dropout)
		classifier_input_size = 64 * widen_factor
		self.classifier_rot = nn.Linear(classifier_input_size, rot_output_size)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNetRot':
		return WideResNetRot(
			depth=args.wrn_depth,
			num_classes=args.nb_classes,
			widen_factor=args.wrn_widen_factor,
			dropout=args.dropout,
			rot_output_size=args.nb_classes_self_supervised,
		)

	def forward_rot(self, x: Tensor) -> Tensor:
		out = self._features(x)
		out = self.classifier_rot(out)
		return out
