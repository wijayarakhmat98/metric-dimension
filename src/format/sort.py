import json
import multiprocessing
import re
import sys
from typing import Any, Dict, Iterator, List
from utils import parse_args

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def format(results : Iterator[str], option_raw : List[str], *args : object, **kwargs : object) -> None:
	option = parse_args(option_raw, [
		(['-h', '--help'], 'not-help', 'true'),
		(['-r', '--regex'], 'r', 'graph'),
		(['-d', '--descending'], 'not-d', 'true')
	])
	if not option['not-help'] or not option['r']:
		usage()
	with multiprocessing.Pool() as pool:
		data = list(pool.imap_unordered(decode, results))
	key = next((key for key in data[0].keys() if re.search(option['r'], key)), None)
	if key:
		data.sort(key=lambda datum: datum[key], reverse=(not option['not-d']))
	for datum in data:
		print(json.dumps(datum))

def usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				... [-r=<pattern>] [-d]

			options:
				-r=<pattern>, --regex=<pattern>
					Sort on the first key that matches with this pattern.
					Defaults to 'graph'.

				-d, --descending
					Sort in descending order.
					Defaults to ascending.
		'''
		).strip(),
		file=sys.stderr
	)
	sys.exit()
