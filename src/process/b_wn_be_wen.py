import json
import metric_dimension
from utils import timer
from typing import Any, cast, Dict

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
	graph = datum['graph']

	with timer() as total_time:
		n, m = metric_dimension.graph6_decode(graph)
		vs = metric_dimension.vertices(n)
		es = metric_dimension.edges(m)

		d = metric_dimension.distance_matrix(m)
		p = metric_dimension.distance_similarity(d)
		with timer() as b_time: b = metric_dimension.find(n, vs, p)
		with timer() as ws_time: ws = metric_dimension.enumerate(n, p, b)
		wn = len(ws)

		de = metric_dimension.distance_matrix_edge(n, d, es)
		pe = metric_dimension.distance_similarity(de)
		with timer() as be_time: be = metric_dimension.find(n, vs, pe)
		with timer() as wes_time: wes = metric_dimension.enumerate(n, pe, be)
		wen = len(wes)

	datum['vertices'] = len(vs)
	datum['metric_dimension'] = b
	datum['metric_dimension_time'] = b_time
	datum['metric_dimension_solutions'] = wn
	datum['metric_dimension_solutions_time'] = ws_time
	datum['edges'] = len(es)
	datum['edge_dimension'] = be
	datum['edge_dimension_time'] = be_time
	datum['edge_dimension_solutions'] = wen
	datum['edge_dimension_solutions_time'] = wes_time
	datum['total_time'] = total_time

	return datum

def encode(datum : Dict[str, Any]) -> str:
	result = json.dumps(datum, default=float)
	return result
