from functools import partial
import json
import multiprocessing
import re
import sys
from typing import Any, Callable, Dict, List, Union
from utils import parse_args

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

def decode(result : str, r : str, format_number : Callable[[float], Union[float, str]]) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	for key, value in datum.items():
		if re.search(r, key):
			datum[key] = format_number(value)
	return datum

def format(results : List[str], sort : bool, option_raw : List[str], *args : object, **kwargs : object) -> None:
	option = parse_args(option_raw, [
		(['-h', '--help'], 'not-help', 'true'),
		(['-r', '--regex'], 'r', ''),
		(['-o', '--magnitude'], 'o', '0'),
		(['-p', '--precision'], 'p', '3'),
		(['-u', '--unit'], 'u', '')
	])
	if not option['not-help'] or not option['r']:
		usage()
	bind_format_number = partial(format_number, o=int(option['o']), p=int(option['p']), u=option['u'])
	bind_decode = partial(decode, r=option['r'], format_number=bind_format_number)
	with multiprocessing.Pool() as pool:
		data = list(pool.imap_unordered(bind_decode, results))
	if sort:
		data.sort(key=lambda datum: datum['graph'])
	for datum in data:
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
		).strip()
	)
	sys.exit()
