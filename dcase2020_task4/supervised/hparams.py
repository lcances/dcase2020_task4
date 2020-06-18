from easydict import EasyDict as edict


def default_supervised_hparams() -> edict:
	# TODO : clean
	hparams = edict()
	hparams.batch_size_s = 64
	hparams.batch_size_u = 64

	hparams.lr = 3e-3
	hparams.weight_decay = 0.0

	# Other param defined for running methods
	hparams.train_name = "Supervised"
	hparams.nb_epochs = 100
	hparams.begin_date = ""
	hparams.dataset_name = ""
	hparams.model_name = ""
	hparams.logdir = ""
	return hparams
