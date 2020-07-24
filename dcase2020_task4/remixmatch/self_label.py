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

	def get_nb_classes(self) -> int:
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

	def get_nb_classes(self) -> int:
		return len(self.angles)


class SelfSupervisedFlips(SelfSupervisedABC):
	def __init__(self):
		self.transforms = [
			lambda x: x,
			lambda x: HorizontalFlip(1.0)(x),
			lambda x: VerticalFlip(1.0)(x),
			lambda x: HorizontalFlip(1.0)(VerticalFlip(1.0)(x)),
		]

	def create_batch_label(self, batch: Tensor) -> (Tensor, Tensor):
		labels = np.random.randint(0, self.get_nb_classes(), len(batch))

		batch = batch.cpu().numpy()
		batch_flipped = torch.as_tensor([
			self.transforms[idx](x) for x, idx in zip(batch, labels)
		]).cuda()

		labels = torch.from_numpy(labels)
		labels = one_hot(labels, 4).float().cuda()

		return batch_flipped, labels

	def get_nb_classes(self) -> int:
		return len(self.transforms)


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
