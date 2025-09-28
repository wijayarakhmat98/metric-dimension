import json
from typing import Any, cast, Dict, Optional, Tuple

preserve_order = True

def header(datum : Dict[str, Any]) -> str:
	return ','.join(datum.keys())

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		datum = cast(Dict[str, Any], json.loads(result))
		return datum
	except:
		return None

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	for key, value in datum.items():
		datum[key] = str(value)
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return ','.join(datum.values())

option_spec = None
option_valid = None
option_augment = None

def help() -> str:
	return ''
