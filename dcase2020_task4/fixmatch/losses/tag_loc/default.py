import torch

from dcase2020_task4.fixmatch.losses.tag_loc.v2 import FixMatchLossMultiHotLocV2

FixMatchLossMultiHotLoc = FixMatchLossMultiHotLocV2


def test():
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

	loss = FixMatchLossMultiHotLoc()

	loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong = loss(
		s_pred_weak_augm_weak,
		s_labels_weak,
		u_pred_weak_augm_weak,
		u_pred_weak_augm_strong,
		u_labels_weak_guessed,
		s_pred_strong_augm_weak,
		s_labels_strong,
		u_pred_strong_augm_weak,
		u_pred_strong_augm_strong,
		u_labels_strong_guessed,
	)

	print("DEBUG: ", loss.shape, loss_s_weak.shape, loss_u_weak.shape, loss_s_strong.shape, loss_u_strong.shape)
	print("DEBUG: ", loss, loss_s_weak, loss_u_weak, loss_s_strong, loss_u_strong)


if __name__ == "__main__":
	test()
