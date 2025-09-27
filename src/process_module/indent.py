import json
from typing import Any, cast, Dict, Tuple

preserve_order = True
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any], option : Tuple[Any, ...]) -> Dict[str, Any]:
	return datum

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum, indent=2)

option_spec = None
option_valid = None

def help() -> str:
	return ''
