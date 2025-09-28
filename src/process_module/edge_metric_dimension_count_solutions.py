import json
import metric_dimension
from utils import timer, parse_switch
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
	bruteforce, = cast(Tuple[bool], option)
	graph = datum['graph']
	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	es = metric_dimension.edges(m)
	d = metric_dimension.distance_matrix(m)
	de = metric_dimension.distance_matrix_edge(n, d, es)
	pe = metric_dimension.distance_similarity(de)
	if 'edge_metric_dimension' in graph:
		be = cast(int, graph['edge_metric_dimension'])
	else:
		if bruteforce:
			be = metric_dimension.find_bruteforce(n, pe)
		else:
			be = metric_dimension.find(n, vs, pe)
	with timer() as wes_time: wes = metric_dimension.enumerate(n, pe, be)
	wen = len(wes)
	datum['edge_metric_dimension_solutions_count'] = wen
	datum['edge_metric_dimension_solutions_count_time'] = wes_time
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	result = json.dumps(datum, default=float)
	return result

option_spec = [
	(['--bruteforce'], False, parse_switch)
]

option_valid = None
option_augment = None

def help() -> str:
	return ''
