import json
from typing import Any, cast, Dict, Optional, Tuple

preserve_order = True
header = None

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		datum = cast(Dict[str, Any], json.loads(result))
		return datum
	except:
		return None

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum, indent=2)

option_spec = None
option_valid = None
option_augment = None

def help() -> str:
	return ''
