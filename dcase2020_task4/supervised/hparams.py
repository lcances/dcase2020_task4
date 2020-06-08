from easydict import EasyDict as edict


def default_supervised_hparams() -> edict:
	hparams = edict()
	hparams.batch_size = 16
	hparams.lr = 1e-2
	hparams.weight_decay = 1e-4

	# Other param defined for running methods
	hparams.nb_epochs = 100
	hparams.begin_date = ""
	hparams.dataset_name = ""
	hparams.model_name = ""
	hparams.logdir = ""
	return hparams
