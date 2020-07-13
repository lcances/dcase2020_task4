from dcase2020_task4.util.utils_standalone import get_model_from_name


def main():
	model = get_model_from_name("CNN03Rot")
	print(model)
	model = model()
	print(model)


if __name__ == "__main__":
	main()
