import json
from typing import Any, Callable, Dict, Tuple

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	return {
		'graph': result
	}

def make_transform(option : Tuple[Any, ...]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
	def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
		return datum
	return transform

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum)

option_spec = None
option_valid = None

def help() -> str:
	return ''
