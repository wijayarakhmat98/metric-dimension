from functools import partial
import json
import multiprocessing
import re
import sys
from typing import Any, Callable, cast, Dict, Iterator, List, Tuple, Union
from utils import parse_args, parse_switch

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

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def decode_then_format_number(result : str, r : str, format_number : Callable[[float], Union[float, str]]) -> Dict[str, Any]:
	datum = decode(result)
	for key, value in datum.items():
		if re.search(r, key):
			datum[key] = format_number(value)
	return datum

def format(results : Iterator[str], option_raw : List[str], *args : object, **kwargs : object) -> None:
	help, regex, magnitude, precision, unit = cast(
		Tuple[bool, str, int, int, str],
		parse_args(option_raw, [
			(['-h', '--help'], False, parse_switch),
			(['-r', '--regex'], '', None),
			(['-o', '--magnitude'], 0, int),
			(['-p', '--precision'], 3, int),
			(['-u', '--unit'], '', None)
		])
	)
	if help or not regex:
		usage()
	bind_format_number = partial(format_number, o=magnitude, p=precision, u=unit)
	bind_decode_then_format_number = partial(decode_then_format_number, r=regex, format_number=bind_format_number)
	with multiprocessing.Pool() as pool:
		for datum in pool.imap_unordered(bind_decode_then_format_number, results):
			print(json.dumps(datum))

def usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
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
		).strip(),
		file=sys.stderr
	)
	sys.exit()
