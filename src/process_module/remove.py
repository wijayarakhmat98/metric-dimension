import json
import re
from typing import Any, cast, Dict, Tuple

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any], option : Tuple[Any, ...]) -> Dict[str, Any]:
	regex, = cast(Tuple[str], option)
	keys_del = [key for key in datum if re.search(regex, key)]
	for key in keys_del:
		del datum[key]
	return datum

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum)

option_spec = [
	(['-r', '--regex'], '', None)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	regex, = cast(Tuple[str], option)
	if not regex:
		return False
	return True

def help() -> str:
	return re.sub(r'\n\t\t\t', r'\n',
	'''
		usage:
			... <-r=<pattern>>

		options:
			-r=<pattern>, --regex=<pattern>
				Remove keys that matches with this pattern.
	'''
	).strip()
