import torch

from dcase2020_task4.fixmatch.losses.loc.v1 import FixMatchLossMultiHotLocV1
from dcase2020_task4.supervised.losses.loc import weak_synth_loss, SupervisedLossLoc


def test_su():
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


def test_fm_loc():
	batch_size = 16
	nb_classes = 10
	audio_size = 431

	s_pred_weak_augm_weak = torch.zeros(batch_size, nb_classes)
	s_labels_weak = torch.zeros(batch_size, nb_classes)
	u_pred_weak_augm_weak = torch.zeros(batch_size, nb_classes)
	u_pred_weak_augm_strong = torch.zeros(batch_size, nb_classes)
	u_labels_weak_guessed = torch.zeros(batch_size, nb_classes)

	s_pred_strong_augm_weak = torch.zeros(batch_size, nb_classes, audio_size)
	s_labels_strong = torch.zeros(batch_size, nb_classes, audio_size)
	u_pred_strong_augm_weak = torch.zeros(batch_size, nb_classes, audio_size)
	u_pred_strong_augm_strong = torch.zeros(batch_size, nb_classes, audio_size)
	u_labels_strong_guessed = torch.zeros(batch_size, nb_classes, audio_size)

	loss = FixMatchLossMultiHotLocV1()

	loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong = loss(
		s_pred_weak_augm_weak, s_labels_weak,
		u_pred_weak_augm_weak, u_pred_weak_augm_strong, u_labels_weak_guessed,
		s_pred_strong_augm_weak, s_labels_strong,
		u_pred_strong_augm_weak, u_pred_strong_augm_strong, u_labels_strong_guessed,
	)

	print("DEBUG: ", loss.shape, loss_s_weak.shape, loss_u_weak.shape, loss_s_strong.shape, loss_u_strong.shape)
	print("DEBUG: ", loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong)


if __name__ == "__main__":
	test()


if __name__ == "__main__":
	test()
