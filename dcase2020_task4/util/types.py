from typing import Optional


def str_to_bool(x: str) -> bool:
	return str(x).lower() in ["true", "1", "yes", "y"]


def str_to_optional_str(x: str) -> Optional[str]:
	if str(x).lower() == "none":
		return None
	else:
		return str(x)
