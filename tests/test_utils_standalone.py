import json

from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2, Noise
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip, RandomTimeDropout, RandomFreqDropout
from augmentation_utils.spec_augmentations import Noise as NoiseSpec
from torchvision.transforms import RandomChoice, Compose
from dcase2020_task4.util.other_spec_augments import CutOutSpec
from dcase2020_task4.util.rand_augment import RandAugment
from dcase2020_task4.util.utils_standalone import get_model_from_name, to_dict_rec


def test_model():
	model = get_model_from_name("CNN03Rot")
	print(model)
	model = model()
	print(model)


def main():
	ratio_augm_weak = 0.5
	augm_list_weak = [
		HorizontalFlip(ratio_augm_weak),
		Occlusion(ratio_augm_weak, max_size=1.0),
	]
	ratio_augm_strong = 1.0
	augm_list_strong = [
		TimeStretch(ratio_augm_strong),
		# PitchShiftRandom(ratio_augm_strong, steps=(-1, 1)),
		Noise(ratio_augm_strong, target_snr=15),
		CutOutSpec(ratio_augm_strong, rect_width_scale_range=(0.1, 0.25), rect_height_scale_range=(0.1, 0.25)),
		RandomTimeDropout(ratio_augm_strong, dropout=0.01),
		RandomFreqDropout(ratio_augm_strong, dropout=0.01),
		NoiseSpec(ratio_augm_strong, snr=5.0),
		Noise2(ratio_augm_strong, noise_factor=(0.005, 0.005)),
	]

	dic = to_dict_rec(augm_list_strong)
	print(dic)
	print(json.dumps(dic, indent="\t"))


if __name__ == "__main__":
	main()
