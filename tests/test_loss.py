import torch

from dcase2020_task4.supervised.loss import weak_synth_loss, SupervisedLossLoc


def test():
	batch_size = 4

	logits_weak = torch.rand(batch_size, 10)
	logits_strong = torch.rand(batch_size, 10, 20)
	y_weak = torch.rand(batch_size, 10)
	y_strong = torch.rand(batch_size, 10, 20)

	pred_weak = logits_weak.sigmoid()
	pred_strong = logits_strong.sigmoid()

	crit_1 = weak_synth_loss
	crit_2 = SupervisedLossLoc()

	loss_w_1, loss_s_1, loss_1 = crit_1(logits_weak, logits_strong, y_weak, y_strong)
	loss_2, loss_w_2, loss_s_2 = crit_2(pred_weak, y_weak, pred_strong, y_strong)

	if loss_1 != loss_2:
		print(loss_1)
		print(loss_2)
		raise RuntimeError("Loss diff %s" % str(loss_1 - loss_2))

	if loss_w_1 != loss_w_2:
		print(loss_w_1)
		print(loss_w_2)
		raise RuntimeError("Loss weak diff %s" % str(loss_w_1 - loss_w_2))

	if loss_s_1 != loss_s_2:
		print(loss_s_1)
		print(loss_s_2)
		raise RuntimeError("Loss strong diff %s" % str(loss_s_1 - loss_s_2))

	print("OK")


if __name__ == "__main__":
	test()
