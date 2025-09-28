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
	n, m = metric_dimension.graph6_decode(graph)
	d = metric_dimension.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)
	if 'metric_dimension' in graph:
		b = cast(int, graph['metric_dimension'])
	else:
		b = metric_dimension.find_bruteforce(n, p)
	with timer() as ws_time: ws = metric_dimension.enumerate(n, p, b)
	wn = len(ws)
	datum['metric_dimension_solutions_count'] = wn
	datum['metric_dimension_solutions_count_time'] = ws_time
	return datum

def encode(datum : Dict[str, Any]) -> str:
	result = json.dumps(datum, default=float)
	return result

option_spec = None
option_valid = None

def help() -> str:
	return ''
