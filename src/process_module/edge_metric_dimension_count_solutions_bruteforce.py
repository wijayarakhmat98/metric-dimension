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
	es = metric_dimension.edges(m)
	d = metric_dimension.distance_matrix(m)
	de = metric_dimension.distance_matrix_edge(n, d, es)
	pe = metric_dimension.distance_similarity(de)
	if 'edge_metric_dimension' in graph:
		be = cast(int, graph['edge_metric_dimension'])
	else:
		be = metric_dimension.find_bruteforce(n, pe)
	with timer() as wes_time: wes = metric_dimension.enumerate(n, pe, be)
	wen = len(wes)
	datum['edge_metric_dimension_solutions_count'] = wen
	datum['edge_metric_dimension_solutions_count_time'] = wes_time
	return datum

def encode(datum : Dict[str, Any]) -> str:
	result = json.dumps(datum, default=float)
	return result

option_spec = None
option_valid = None

def help() -> str:
	return ''
