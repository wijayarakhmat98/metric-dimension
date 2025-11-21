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
	key, value = cast(Tuple[Optional[str], Optional[str]], option)
	if not datum:
		return None
	if key is not None:
		datum[key] = value
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum)

option_spec = [
	(['--key'], None, None),
	(['--value'], None, None)
]

option_valid = None
option_augment = None

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		usage:
			... [--key=<string> [--value=<string>]]

		options:
			--key=<string>
				Add a new key.

			--value=<string>
				The value for the new key.
				None by default, a string otherwise.
	'''
	).strip()
