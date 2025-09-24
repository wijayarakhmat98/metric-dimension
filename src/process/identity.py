import json
from typing import Any, Dict

def decode(result : str) -> Dict[str, Any]:
	return {
		'graph': result
	}

def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
	return datum

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum)
