import json
import re
from typing import Any, cast, Dict, Optional, Tuple

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
	regex, = cast(Tuple[str], option)
	keys_del = [key for key in datum if re.search(regex, key)]
	for key in keys_del:
		del datum[key]
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum)

option_spec = [
	(['-r', '--regex'], '', None)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	regex, = cast(Tuple[str], option)
	if not regex:
		return False
	return True

option_augment = None

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
