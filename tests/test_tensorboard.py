import torch
from torch.utils.tensorboard import SummaryWriter


def main():
	dirpath = "../../tensorboard/tests"
	writer = SummaryWriter(log_dir=dirpath, comment="TEST")

	print("Save distributions...")
	distributions = torch.as_tensor([10, 20, 5, 10])
	writer.add_histogram("hist_4", distributions, bins="auto")
	for v in distributions:
		writer.add_hparams({}, {"acc": v})

	writer.close()


if __name__ == "__main__":
	main()
