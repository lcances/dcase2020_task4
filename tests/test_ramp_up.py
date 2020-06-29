from dcase2020_task4.util.ramp_up import RampUp


def test():
	rampup = RampUp(1.0, 10)
	for i in range(15):
		print("Value:", rampup.value())
		rampup.step()


if __name__ == "__main__":
	test()
