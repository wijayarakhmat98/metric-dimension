import json
from typing import Any, cast, Dict, List, Optional, Tuple

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
		datum['MD_algorithm_time'] = datum.pop('metric_dimension_total_time')
		datum['MD_search_time'] = datum.pop('metric_dimension_time')
		nV = cast(int, datum['vertices'])
		internal_time = cast(List[float], datum.pop('metric_dimension_internal_time'))
		N = len(internal_time)
		k = nV - N + 1
		k0_time = internal_time[-1] if N >= 1 else None
		k1_time = internal_time[-2] if N >= 2 else None
		datum['MD_k'] = k
		datum['MD_k_time'] = k0_time
		datum['MD_k+1_time'] = k1_time
	if 'edge_metric_dimension' in datum:
		EMD = cast(int, datum['edge_metric_dimension'])
		if EMD < 0:
			datum['edge_metric_dimension'] = None
		datum['EMD_algorithm_time'] = datum.pop('edge_metric_dimension_total_time')
		datum['EMD_search_time'] = datum.pop('edge_metric_dimension_time')
		nV = cast(int, datum['vertices'])
		internal_time = cast(List[float], datum.pop('edge_metric_dimension_internal_time'))
		N = len(internal_time)
		k = nV - N + 1
		k0_time = internal_time[-1] if N >= 1 else None
		k1_time = internal_time[-2] if N >= 2 else None
		datum['EMD_k'] = k
		datum['EMD_k_time'] = k0_time
		datum['EMD_k+1_time'] = k1_time
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
