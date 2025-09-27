import json
from typing import Any, Dict, Tuple

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	return {
		'graph': result
	}

def transform(datum : Dict[str, Any], option : Tuple[Any, ...]) -> Dict[str, Any]:
	return datum

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum)

option_spec = None
option_valid = None

def help() -> str:
	return ''
