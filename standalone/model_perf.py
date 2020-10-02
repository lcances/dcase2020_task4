import numpy as np
import torch

from argparse import ArgumentParser, Namespace
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose

from dcase2020_task4.util.datasets.onehot_dataset import OneHotDataset
from dcase2020_task4.util.other_metrics import CategoricalAccuracyOnehot
from dcase2020_task4.util.utils_standalone import build_model_from_args


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--dataset_name", type=str, default="CIFAR10", choices=["UBS8K", "CIFAR10"])
	parser.add_argument("--dataset_path", type=str, default="/projets/samova/leocances/CIFAR10/")
	parser.add_argument("--nb_classes", type=int, default=10)
	parser.add_argument("--seed", type=int, default=123)

	parser.add_argument("--batch_size_s", type=int, default=64)
	parser.add_argument("--model", type=str, default="WideResNet28", help="Model name.")
	parser.add_argument("--filepath_model", type=str, default=None, help="Path to the torch data of the model.")

	return parser.parse_args()


def main():
	args = create_args()
	dataset_val = get_validation_dataset(args)
	dataset_val = OneHotDataset(dataset_val, args.nb_classes)

	model = get_model(args)
	acti_fn = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=True)

	with torch.no_grad():
		metric = CategoricalAccuracyOnehot(dim=1)
		conf_matrix = torch.zeros(args.nb_classes, args.nb_classes)
		mean = None

		metric.reset()
		model.eval()

		iter_val = iter(loader_val)
		for i, (x_batch, x_label) in enumerate(iter_val):
			x_batch = x_batch.cuda().float()
			x_label = x_label.cuda().float()

			x_logits = model(x_batch)
			x_pred = acti_fn(x_logits, dim=1)

			class_label = x_label.argmax(dim=1)
			class_pred = x_pred.argmax(dim=1)

			conf_matrix[class_label][class_pred] += 1
			mean = metric(x_pred, x_label)

		print("Global mean score : ", mean)
		print("Conf matrix : \n", conf_matrix)


def get_validation_dataset(args: Namespace) -> Dataset:
	pre_process_fn = lambda img: img
	post_process_fn = Compose([
		ToTensor(),
		transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0),
	])
	transforms_val = [pre_process_fn, post_process_fn]
	dataset_val = CIFAR10(args.dataset_path, train=False, download=True, transform=Compose(transforms_val))
	return dataset_val


def get_model(args: Namespace) -> Module:
	state_dict = torch.load(args.filepath_model)
	model = build_model_from_args(args)
	model.load_state_dict(state_dict["state_dict"])
	return model


if __name__ == "__main__":
	main()
