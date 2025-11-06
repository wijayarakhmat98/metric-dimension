import json
import metric_dimension
from utils import timer
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
	s = datum['graph']
	M = metric_dimension.graph6_decode(s)
	DV = metric_dimension.distance_matrix(M)
	D = DV
	B = metric_dimension.distance_similarity(D)
	P = metric_dimension.reduced_distance_similarity(B)
	with timer() as k_time: k = metric_dimension.find_pseudo_boolean(P)
	datum['metric_dimension'] = k
	datum['metric_dimension_time'] = k_time
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	result = json.dumps(datum, default=float)
	return result

option_spec = None
option_valid = None
option_augment = None

def help() -> str:
	return ''
