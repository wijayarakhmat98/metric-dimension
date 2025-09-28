import json
import metric_dimension
from typing import Any, Dict, Optional, Tuple

preserve_order = False
header = None

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		graph = result
		_, _ = metric_dimension.graph6_decode(graph)
		return {
			'graph': result
		}
	except:
		return None

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum)

option_spec = None
option_valid = None
option_augment = None

def help() -> str:
	return ''
