from dcase2020_task4.util.FnDataLoader import FnDataLoader


class NoLabelDataLoader(FnDataLoader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, fn=lambda x, y: [x], **kwargs)
