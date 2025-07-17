import ast
import metric_dimension
from typing import Any, Dict
from time import perf_counter

def process(graph : str) -> str:
	time = perf_counter()

	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	es = metric_dimension.edges(m)

	d = metric_dimension.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)

	b_time = perf_counter()
	b = metric_dimension.find(vs, p)
	b_time = perf_counter() - b_time

	ws_time = perf_counter()
	_ws = metric_dimension.enumerate(vs, p, b)
	ws_time = perf_counter() - ws_time

	ws = [[str(v) for v in vs[w]] for w in _ws]
	rs = [metric_dimension.resolving_representation(w, d) for w in _ws]
	rs_valid = [metric_dimension.is_resolving_valid(r) for r in rs]
	wss = [{'set': w, 'is_sane': r_valid} for w, r_valid in zip(ws, rs_valid)]

	de = metric_dimension.distance_matrix_edge(n, d, es)
	pe = metric_dimension.distance_similarity(de)

	be_time = perf_counter()
	be = metric_dimension.find(vs, pe)
	be_time = perf_counter() - be_time

	wes_time = perf_counter()
	_wes = metric_dimension.enumerate(vs, pe, be)
	wes_time = perf_counter() - wes_time

	wes = [[str(v) for v in vs[we]] for we in _wes]
	res = [metric_dimension.resolving_representation(we, de) for we in _wes]
	res_valid = [metric_dimension.is_resolving_valid(re) for re in res]
	wess = [{'set': we, 'is_sane': re_valid} for we, re_valid in zip(wes, res_valid)]

	time = perf_counter() - time

	return str({
		'graph': graph,
		'vertex': {
			'dimension': b,
			'time': b_time,
			'enumerate': {
				'n': len(ws),
				'resolving': wss,
				'time': ws_time
			}
		},
		'edge': {
			'dimension': be,
			'time': be_time,
			'enumerate': {
				'n': len(wes),
				'resolving': wess,
				'time': wes_time
			}
		},
		'time': time
	})

def resume(s : str) -> str:
	data : Dict[str, Any] = ast.literal_eval(s)
	graph : str = data['graph']
	return graph
