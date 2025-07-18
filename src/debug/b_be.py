import json
import metric_dimension
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar('T')

class timer():
	def __enter__(self) -> 'timer':
		self.start = perf_counter()
		self.elapsed = 0.0
		return self

	def __exit__(self, *_ : Any) -> None:
		self.end = perf_counter()
		self.elapsed = self.end - self.start

	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * self.elapsed, 3))

def debug(graph : str) -> None:
	with timer() as total_time:
		n, m = metric_dimension.graph6_decode(graph)
		vs = metric_dimension.vertices(n)
		es = metric_dimension.edges(m)

		d = metric_dimension.distance_matrix(m)
		p = metric_dimension.distance_similarity(d)
		with timer() as b_time: b = metric_dimension.find(vs, p)

		de = metric_dimension.distance_matrix_edge(n, d, es)
		pe = metric_dimension.distance_similarity(de)
		with timer() as be_time: be = metric_dimension.find(vs, pe)

	print(json.dumps({
		'graph': graph,
		'vertices': len(vs),
		'metric_dimension': b,
		'metric_dimension_time': b_time,
		'edges': len(es),
		'edge_dimension': be,
		'edge_dimension_time': be_time,
		'total_time': total_time
	}, default=str, indent=2))
