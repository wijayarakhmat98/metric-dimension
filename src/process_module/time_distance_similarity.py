import json
import metric_dimension
from utils import timer
from typing import Any, cast, Dict, Tuple

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any], option : Tuple[Any, ...]) -> Dict[str, Any]:
	graph = datum['graph']
	_, m = metric_dimension.graph6_decode(graph)
	d = metric_dimension.distance_matrix(m)
	with timer() as p_time: _ = metric_dimension.distance_similarity(d)
	datum['distance_similiraty_time'] = p_time
	return datum

def encode(datum : Dict[str, Any]) -> str:
	result = json.dumps(datum, default=float)
	return result

option_spec = None
option_valid = None

def help() -> str:
	return ''
