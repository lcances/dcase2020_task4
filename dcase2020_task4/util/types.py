from typing import Optional, Union


def str_to_bool(x: str) -> bool:
	x_low = str(x).lower()
	if x_low in ["true", "1", "yes", "y"]:
		return True
	elif x_low in ["false", "0", "no", "n"]:
		return False
	else:
		raise RuntimeError("Invalid boolean argument \"%s\"." % x)


def str_to_optional_str(x: str) -> Optional[str]:
	x = str(x)
	if x.lower() == "none":
		return None
	else:
		return x


def str_to_optional_int(x: str) -> Optional[int]:
	x = str(x)
	if x.lower() == "none":
		return None
	else:
		return int(x)


def str_to_union_str_int(x: str) -> Union[str, int]:
	x = str(x)
	if x.isdigit():
		return int(x)
	else:
		return x
