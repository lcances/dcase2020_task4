from typing import Optional, Union


def str_to_bool(x: str) -> bool:
	x = str(x)
	return x.lower() in ["true", "1", "yes", "y"]


def str_to_optional_str(x: str) -> Optional[str]:
	x = str(x)
	if x.lower() == "none":
		return None
	else:
		return x


def str_to_union_str_int(x: str) -> Union[str, int]:
	x = str(x)
	if x.isdigit():
		return int(x)
	else:
		return x
