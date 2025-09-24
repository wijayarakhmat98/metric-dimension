import json
from typing import Any, cast, Dict

preserve_order = True

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
	return datum

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum, indent=2)
