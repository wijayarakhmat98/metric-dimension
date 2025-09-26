import json
import metric_dimension
from utils import timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, *args : object, **kwargs : object) -> None:
	with timer_ms() as total_time:
		n, m = metric_dimension.graph6_decode(graph)
		vs = metric_dimension.vertices(n)

		d = metric_dimension.distance_matrix(m)
		p = metric_dimension.distance_similarity(d)
		with timer_ms() as b_time: b = metric_dimension.find_bruteforce(n, p)

	print(json.dumps({
		'graph': graph,
		'vertices': len(vs),
		'metric_dimension': b,
		'metric_dimension_time': b_time,
		'total_time': total_time
	}, default=str, indent=2))
