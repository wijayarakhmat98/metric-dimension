import json
import metric_dimension
from utils import timer, parse_switch
from typing import Any, cast, Dict, Tuple

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any], option : Tuple[Any, ...]) -> Dict[str, Any]:
	bruteforce, = cast(Tuple[bool], option)
	graph = datum['graph']
	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	es = metric_dimension.edges(m)
	d = metric_dimension.distance_matrix(m)
	de = metric_dimension.distance_matrix_edge(n, d, es)
	pe = metric_dimension.distance_similarity(de)
	with timer() as be_time:
		if bruteforce:
			be = metric_dimension.find_bruteforce(n, pe)
		else:
			be = metric_dimension.find(n, vs, pe)
	datum['edge_metric_dimension'] = be
	datum['edge_metric_dimension_time'] = be_time
	return datum

def encode(datum : Dict[str, Any]) -> str:
	result = json.dumps(datum, default=float)
	return result

option_spec = [
	(['--bruteforce'], False, parse_switch)
]

option_valid = None

def help() -> str:
	return ''
