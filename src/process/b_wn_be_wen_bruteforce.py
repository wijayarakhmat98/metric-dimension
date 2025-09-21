import json
import metric_dimension
from utils import timer

def process(graph : str, *args : object, **kwargs : object) -> str:
	with timer() as total_time:
		n, m = metric_dimension.graph6_decode(graph)
		vs = metric_dimension.vertices(n)
		es = metric_dimension.edges(m)

		d = metric_dimension.distance_matrix(m)
		p = metric_dimension.distance_similarity(d)
		with timer() as b_time: b = metric_dimension.find_bruteforce(n, p)
		with timer() as ws_time: ws = metric_dimension.enumerate(n, p, b)
		wn = len(ws)

		de = metric_dimension.distance_matrix_edge(n, d, es)
		pe = metric_dimension.distance_similarity(de)
		with timer() as be_time: be = metric_dimension.find_bruteforce(n, pe)
		with timer() as wes_time: wes = metric_dimension.enumerate(n, pe, be)
		wen = len(wes)

	return json.dumps({
		'graph': graph,
		'vertices': len(vs),
		'metric_dimension': b,
		'metric_dimension_time': b_time,
		'metric_dimension_solutions': wn,
		'metric_dimension_solutions_time': ws_time,
		'edges': len(es),
		'edge_dimension': be,
		'edge_dimension_time': be_time,
		'edge_dimension_solutions': wen,
		'edge_dimension_solutions_time': wes_time,
		'total_time': total_time
	}, default=float)
