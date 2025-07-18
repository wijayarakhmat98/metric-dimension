from functools import partial
import json
import multiprocessing
import re
import sys
from typing import Any, Dict, List
from utils import parse_args

def decode(result : str, r : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	keys_del = [key for key in datum if re.search(r, key)]
	for key in keys_del:
		if re.search(r, key):
			del datum[key]
	return datum

def format(results : List[str], sort : bool, option_raw : List[str], *args : object, **kwargs : object) -> None:
	option = parse_args(option_raw, [
		(['-h', '--help'], 'not-help', 'true'),
		(['-r', '--regex'], 'r', ''),
	])
	if not option['not-help'] or not option['r']:
		usage()
	bind_decode = partial(decode, r=option['r'])
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
				... <-r=<pattern>>

			options:
				-r=<pattern>, --regex=<pattern>
				 Remove keys that matches with this pattern.
		'''
		).strip()
	)
	sys.exit()
