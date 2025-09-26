import json
from typing import Any, Callable, cast, Dict, Tuple

preserve_order = True

def header(datum : Dict[str, Any]) -> str:
	return ','.join(datum.keys())

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def make_transform(option : Tuple[Any, ...]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
	def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
		for key, value in datum.items():
			datum[key] = str(value)
		return datum
	return transform

def encode(datum : Dict[str, Any]) -> str:
	return ','.join(datum.values())

option_spec = None
option_valid = None

def help() -> str:
	return ''
