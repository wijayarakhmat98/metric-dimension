import json
import metric_dimension
from typing import Any, Dict

def process(graph : str) -> str:
	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	es = metric_dimension.edges(m)

	d = metric_dimension.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)
	b = metric_dimension.find(vs, p)

	de = metric_dimension.distance_matrix_edge(n, d, es)
	pe = metric_dimension.distance_similarity(de)
	be = metric_dimension.find(vs, pe)

	return json.dumps({
		'graph': graph,
		'metric_dimension': b,
		'edge_dimension': be
	})

def resume(result : str) -> str:
	datum : Dict[str, Any] = json.loads(result)
	graph : str = datum['graph']
	return graph
