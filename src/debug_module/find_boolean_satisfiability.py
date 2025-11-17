import metric_dimension
import re
from typing import Any, cast, Tuple
from utils import timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option : Tuple[Any, ...]) -> None:
	mode, = cast(Tuple[str], option)
	if mode not in ('metric-dimension', 'edge-metric-dimension'):
		return None

	M = metric_dimension.graph6_decode(graph)
	V = metric_dimension.vertices(M)
	nV = len(V)
	print('Graph: {}'.format(graph))
	print('Vertices: {}'.format(nV))
	print()

	DV = metric_dimension.distance_matrix(M)
	match mode:
		case 'metric-dimension':
			D = DV
		case 'edge-metric-dimension':
			E = metric_dimension.edges(M)
			DE = metric_dimension.edge_distance_matrix(E, DV)
			D = DE

	print('Distance similarity...')
	with timer_ms() as B_time: B = metric_dimension.distance_similarity(D)
	print('Distance similarity time: {}'.format(B_time))
	print()

	print('Reduced distance similarity...')
	with timer_ms() as P_time: P = metric_dimension.reduced_distance_similarity(B)
	print('Reduced distance similarity time: {}'.format(P_time))
	print()

	config = metric_dimension.find_config_boolean_satisfiability(P)

	print('Metric dimension...')
	with timer_ms() as total_time:
		for k in range(nV - 1, -1, -1):
			print('Trying k = {}... '.format(k), end='', flush=True)
			with timer_ms() as k_time:
				found = metric_dimension.find_exact_boolean_satisfiability(config, k)
			print(k_time)
			if not found:
				k = k + 1
				break
		else:
			k = 0
	print('Metric dimension: {}'.format(k))
	print('Metric dimension time: {}'.format(total_time))
	print()

option_spec = [
	(['-f', '--find'], 'metric-dimension', None)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	mode, = cast(Tuple[str], option)
	if mode not in ('metric-dimension', 'edge-metric-dimension'):
		return False
	return True

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		options:
			-f=<mode>, --find=<mode>
				metric-dimension
				edge-metric-dimension
				Defaults to metric-dimension.
	'''
	).strip()
