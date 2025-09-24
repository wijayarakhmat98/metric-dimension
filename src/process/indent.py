import json
from typing import Any, Callable, cast, Dict, Tuple

preserve_order = True
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def make_transform(option : Tuple[Any, ...]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
	def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
		return datum
	return transform

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum, indent=2)

option_spec = None
option_valid = None

def help() -> str:
	return ''
