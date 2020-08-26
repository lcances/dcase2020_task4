from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from dcase2020_task4.other_models.wrn import resnet
from dcase2020_task4.other_models import wrn_utils as utils


class WideResNet(Module):
	def __init__(self, depth: int = 28, width: int = 2, nb_classes: int = 10):
		super().__init__()
		f, params = resnet(depth, width, nb_classes)
		self.f = f
		self.params = params

	def forward(self, x: Tensor) -> Tensor:
		return self.f(x, self.params, self.get_mode())

	def get_mode(self) -> bool:
		return self.training

	def parameters(self, recurse: bool = True) -> list:
		return [v for v in self.params.values() if v.requires_grad]


class WideResNetRot(Module):
	def __init__(self, depth: int = 28, width: int = 2, nb_classes: int = 10, num_rot: int = 4):
		super().__init__()
		f, f_rot, params = WideResNetRot.resnet_rot(depth, width, nb_classes, num_rot)
		self.f = f
		self.f_rot = f_rot
		self.params = params

	def forward_rot(self, x: Tensor) -> Tensor:
		return self.f_rot(x, self.params, self.get_mode())

	@staticmethod
	def resnet_rot(depth: int, width: int, num_classes: int, num_rot: int):
		assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
		n = (depth - 4) // 6
		widths = [int(v * width) for v in (16, 32, 64)]

		def gen_block_params(ni, no):
			return {
				'conv0': utils.conv_params(ni, no, 3),
				'conv1': utils.conv_params(no, no, 3),
				'bn0': utils.bnparams(ni),
				'bn1': utils.bnparams(no),
				'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
			}

		def gen_group_params(ni, no, count):
			return {'block%d' % i: gen_block_params(ni if i == 0 else no, no) for i in range(count)}

		flat_params = utils.cast(utils.flatten({
			'conv0': utils.conv_params(3, 16, 3),
			'group0': gen_group_params(16, widths[0], n),
			'group1': gen_group_params(widths[0], widths[1], n),
			'group2': gen_group_params(widths[1], widths[2], n),
			'bn': utils.bnparams(widths[2]),
			'fc': utils.linear_params(widths[2], num_classes),
			'fc_rot': utils.linear_params(widths[2], num_rot),
		}))

		utils.set_requires_grad_except_bn_(flat_params)

		def block(x, params, base, mode, stride):
			o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
			y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
			o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
			z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
			if base + '.convdim' in params:
				return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
			else:
				return z + x

		def group(o, params, base, mode, stride):
			for i in range(n):
				o = block(o, params, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
			return o

		def f(input, params, mode):
			x = F.conv2d(input, params['conv0'], padding=1)
			g0 = group(x, params, 'group0', mode, 1)
			g1 = group(g0, params, 'group1', mode, 2)
			g2 = group(g1, params, 'group2', mode, 2)
			o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
			o = F.avg_pool2d(o, 8, 1, 0)
			o = o.view(o.size(0), -1)
			o = F.linear(o, params['fc.weight'], params['fc.bias'])
			return o

		def f_rot(input, params, mode):
			x = F.conv2d(input, params['conv0'], padding=1)
			g0 = group(x, params, 'group0', mode, 1)
			g1 = group(g0, params, 'group1', mode, 2)
			g2 = group(g1, params, 'group2', mode, 2)
			o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
			o = F.avg_pool2d(o, 8, 1, 0)
			o = o.view(o.size(0), -1)
			o = F.linear(o, params['fc_rot.weight'], params['fc_rot.bias'])
			return o

		return f, f_rot, flat_params

	def get_mode(self) -> bool:
		return self.training

	def parameters(self, recurse: bool = True) -> list:
		return [v for v in self.params.values() if v.requires_grad]


def test():
	from dcase2020_task4.util.utils_standalone import get_nb_parameters, get_nb_trainable_parameters

	wrn = WideResNet()
	print("Nb params   : ", get_nb_parameters(wrn))
	print("Nb trainable: ", get_nb_trainable_parameters(wrn))


if __name__ == "__main__":
	test()
