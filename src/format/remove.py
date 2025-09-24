from functools import partial
import json
import multiprocessing
import re
import sys
from typing import Any, cast, Dict, Iterator, List, Tuple
from utils import parse_args, parse_switch

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def decode_then_remove(result : str, r : str) -> Dict[str, Any]:
	datum = decode(result)
	keys_del = [key for key in datum if re.search(r, key)]
	for key in keys_del:
		if re.search(r, key):
			del datum[key]
	return datum

def format(results : Iterator[str], option_raw : List[str], *args : object, **kwargs : object) -> None:
	help, regex = cast(
		Tuple[bool, str],
		parse_args(option_raw, [
			(['-h', '--help'], False, parse_switch),
			(['-r', '--regex'], '', None),
		])
	)
	if help or not regex:
		usage()
	bind_decode_then_remove = partial(decode_then_remove, r=regex)
	with multiprocessing.Pool() as pool:
		for datum in pool.imap_unordered(bind_decode_then_remove, results):
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
		).strip(),
		file=sys.stderr
	)
	sys.exit()
