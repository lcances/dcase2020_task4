from easydict import EasyDict as edict


def default_mixmatch_hparams() -> edict:
	hparams = edict()
	hparams.batch_size = 16
	hparams.nb_augms = 2
	hparams.sharpen_temp = 0.5
	hparams.mixup_alpha = 0.75
	hparams.lambda_u_max = 10.0  # In paper : 75
	hparams.lr = 1e-2
	hparams.weight_decay = 8e-4
	hparams.criterion_name_u = "sqdiff"  # In paper : sqdiff, can use crossentropy
	return hparams
