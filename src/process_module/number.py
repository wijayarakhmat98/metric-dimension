import json
import re
from typing import Any, cast, Dict, Optional, Tuple, Union

preserve_order = False
header = None

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		datum = cast(Dict[str, Any], json.loads(result))
		return datum
	except:
		return None

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	regex, magnitude, precision, unit = cast(Tuple[str, int, int, str], option)
	for key, value in datum.items():
		if re.search(regex, key):
			datum[key] = format_number(value, magnitude, precision, unit)
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum)

option_spec = [
	(['-r', '--regex'], '', None),
	(['-o', '--magnitude'], 0, int),
	(['-p', '--precision'], 3, int),
	(['-u', '--unit'], '', None)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	regex, _, _, _ = cast(Tuple[str, int, int, str], option)
	if not regex:
		return False
	return True

option_augment = None

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		usage:
			... <-r=<pattern>> [-m=<magnitude>] [-p=<precision>] [-u=<unit>]

		options:
			-r=<pattern>, --regex=<pattern>
				Match on keys with this pattern.

			-o=<magnitude>, --magnitude=<magnitude>
				Multiplies by 10^o.
				Defaults to o = 0.

			-p=<precision>, --precision=<precision>
				Round to p numbers after floating point.
				Defaults to p = 3.

			-u=<unit>, --unit=<unit>
				Add unit after number.
	'''
	).strip()

def format_number(x : float, o : int, p : int, u : str) -> Union[int, float, str]:
	y : Union[int, float]
	if p == 0:
		y = int(10**o * x)
	else:
		y = round(10**o * x, p)
	if u:
		return '{}{}'.format(y, u)
	else:
		return y
