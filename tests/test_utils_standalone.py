import json

from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom, Occlusion, Noise2
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip, Noise, RandomTimeDropout, RandomFreqDropout
from torchvision.transforms import RandomChoice, Compose
from dcase2020_task4.util.rand_augment import RandAugment
from dcase2020_task4.util.utils_standalone import get_model_from_name, augm_fn_to_dict


def test_model():
	model = get_model_from_name("CNN03Rot")
	print(model)
	model = model()
	print(model)


def main():
	ratio = 0.5
	augm_fn = RandomChoice([
		HorizontalFlip(ratio),
		TimeStretch(ratio),
		PitchShiftRandom(ratio, steps=(-1, 1)),
		Noise(ratio=ratio, snr=5.0),
		Noise2(ratio, noise_factor=(5.0, 5.0)),
		RandAugment(),
	])

	dic = augm_fn_to_dict(augm_fn)
	print(dic)
	print(json.dumps(dic, indent="\t"))


if __name__ == "__main__":
	main()
