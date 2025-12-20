import json
from typing import Any, cast, Dict, Optional, Tuple

preserve_order = False
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
	if 'metric_dimension' in datum:
		MD = cast(int, datum['metric_dimension'])
		if MD < 0:
			datum['metric_dimension'] = None
		datum['MD_time'] = datum.pop('metric_dimension_total_time')
		del datum['metric_dimension_time']
		del datum['metric_dimension_internal_time']
	if 'edge_metric_dimension' in datum:
		EMD = cast(int, datum['edge_metric_dimension'])
		if EMD < 0:
			datum['edge_metric_dimension'] = None
		datum['EMD_time'] = datum.pop('edge_metric_dimension_total_time')
		del datum['edge_metric_dimension_time']
		del datum['edge_metric_dimension_internal_time']
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
