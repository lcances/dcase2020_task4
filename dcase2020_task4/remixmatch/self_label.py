import numpy as np
import torch

from abc import ABC
from torch import Tensor
from torch.nn.functional import one_hot

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip


class SelfSupervisedABC(ABC):
	def create_batch_label(self, batch: Tensor) -> (Tensor, Tensor):
		raise NotImplementedError("Abstract method")


class SelfSupervisedRotation(SelfSupervisedABC):
	def __init__(self):
		self.angles = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
		self.rotation_fn = lambda x, ang: Transform(1.0, rotation=(ang, ang))(x)

	def create_batch_label(self, batch: Tensor) -> (Tensor, Tensor):
		labels = np.random.randint(0, len(self.angles), len(batch))
		angles = self.angles[labels]

		batch_rotated = torch.stack([
			self.rotation_fn(x, ang) for x, ang in zip(batch, angles)
		]).cuda()

		labels = torch.from_numpy(labels)
		labels = one_hot(labels, len(self.angles)).float().cuda()

		return batch_rotated, labels


class SelfSupervisedFlips(SelfSupervisedABC):
	def __init__(self):
		self.flip_fn = lambda x, idx: {
			0: x,
			1: HorizontalFlip(1.0)(x),
			2: VerticalFlip(1.0)(x),
			3: HorizontalFlip(1.0)(VerticalFlip(1.0)(x)),
		}[idx]

	def create_batch_label(self, batch: Tensor) -> (Tensor, Tensor):
		labels = np.random.randint(0, 4, len(batch))

		batch = batch.cpu().numpy()
		batch_flipped = torch.as_tensor([
			self.flip_fn(x, idx) for x, idx in zip(batch, labels)
		]).cuda()

		labels = torch.from_numpy(labels)
		labels = one_hot(labels, 4).float().cuda()

		return batch_flipped, labels


def apply_random_rotation(batch: Tensor, angles_allowed) -> (Tensor, Tensor):
	# TODO : rem
	indexes = np.random.randint(0, len(angles_allowed), len(batch))
	angles = angles_allowed[indexes]
	res = torch.stack([
		Transform(1.0, rotation=(ang, ang))(x) for x, ang in zip(batch, angles)
	]).cuda()
	return res, torch.from_numpy(indexes)


def test():
	ss_transform = SelfSupervisedFlips()
	batch = torch.ones(16, 64, 400)

	x = batch[0].numpy()
	augm = HorizontalFlip(1.0)
	x2 = augm(x)
	print(x2.shape)

	batch_flipped, labels = ss_transform.create_batch_label(batch)
	print(batch_flipped.shape)
	print(labels.shape)


if __name__ == "__main__":
	test()
