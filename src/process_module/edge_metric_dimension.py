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
	with timer() as be_time:
		if bruteforce:
			be = metric_dimension.find_bruteforce(n, pe)
		else:
			be = metric_dimension.find(n, vs, pe)
	datum['edge_metric_dimension'] = be
	datum['edge_metric_dimension_time'] = be_time
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
