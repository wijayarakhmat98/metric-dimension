import metric_dimension
from typing import Any, Tuple
from utils import timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option : Tuple[Any, ...]) -> None:
	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	print('Graph: {}'.format(graph))
	print('Vertices: {}'.format(len(vs)))
	print()

	d = metric_dimension.distance_matrix(m)

	print('Distance similarity...')
	with timer_ms() as p_time: p = metric_dimension.distance_similarity(d)
	print('Distance similarity time: {}'.format(p_time))
	print()

	print('Metric dimension...')
	with timer_ms() as b_time:
		for k in range(n - 1, -1, -1):
			print('Trying k = {}... '.format(k), end='', flush=True)
			with timer_ms() as k_time: found = metric_dimension.find_exact(vs, p, k)
			print(k_time)
			if not found:
				b = k + 1
				break
		else:
			b = 0
	print('Metric dimension: {}'.format(b))
	print('Metric dimension time: {}'.format(b_time))
	print()

option_spec = None
option_valid = None

def help() -> str:
	return ''
