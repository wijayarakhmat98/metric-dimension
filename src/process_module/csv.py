import json
from typing import Any, cast, Dict, Tuple

preserve_order = True

def header(datum : Dict[str, Any]) -> str:
	return ','.join(datum.keys())

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any], option : Tuple[Any, ...]) -> Dict[str, Any]:
	for key, value in datum.items():
		datum[key] = str(value)
	return datum

def encode(datum : Dict[str, Any]) -> str:
	return ','.join(datum.values())

option_spec = None
option_valid = None

def help() -> str:
	return ''
