import metric_dimension
import re
from typing import Any, cast, Tuple
from utils import parse_switch, timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option : Tuple[Any, ...]) -> None:
	method, edge = cast(Tuple[str, bool], option)
	if method not in ('bruteforce', 'boolean_satisfiability', 'linear_integer_arithmetic', 'pseudo_boolean'):
		return None

	M = metric_dimension.graph6_decode(graph)
	nV = M.shape[0]
	print('Graph: {}'.format(graph))
	print('Vertices: {}'.format(nV))
	print()

	DV = metric_dimension.distance_matrix(M)
	if edge:
		E = metric_dimension.edges(M)
		DE = metric_dimension.edge_distance_matrix(E, DV)
		D = DE
	else:
		D = DV

	print('Distance similarity...')
	with timer_ms() as B_time: B = metric_dimension.distance_similarity(D)
	print('Distance similarity time: {}'.format(B_time))
	print()

	print('Reduced distance similarity...')
	with timer_ms() as P_time: P = metric_dimension.reduced_distance_similarity(B)
	print('Reduced distance similarity time: {}'.format(P_time))
	print()

	find : metric_dimension.find
	match method:
		case 'bruteforce':
			find = metric_dimension.find_bruteforce(P)
		case 'boolean_satisfiability':
			find = metric_dimension.find_boolean_satisfiability(P)
		case 'linear_integer_arithmetic':
			find = metric_dimension.find_linear_integer_arithmetic(P)
		case 'pseudo_boolean':
			find = metric_dimension.find_pseudo_boolean(P)

	print('Metric dimension...')
	with timer_ms() as total_time:
		for k in range(nV - 1, -1, -1):
			print('Trying k = {}... '.format(k), end='', flush=True)
			with timer_ms() as k_time:
				found = find.exact(k)
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
	(['-a', '--algorithm'], '', None),
	(['--edge'], False, parse_switch)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	method, _ = cast(Tuple[str, bool], option)
	if method not in ('bruteforce', 'boolean_satisfiability', 'linear_integer_arithmetic', 'pseudo_boolean'):
		return False
	return True

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		options:
			-a=<method>, --algorithm=<method>
				bruteforce
				boolean_satisfiability
				linear_integer_arithmetic
				pseudo_boolean

			--edge
				Find edge metric dimension instead of metric dimension.
	'''
	).strip()
