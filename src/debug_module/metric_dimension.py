import metric_dimension
import re
from typing import Any, cast, Tuple
from utils import parse_switch, timer

class timer_ms(timer):
	def __init__(self, time : None | timer = None) -> None:
		if time:
			self.elapsed = time.elapsed

	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option : Tuple[Any, ...]) -> None:
	method, edge, limit = cast(Tuple[str, bool, float], option)
	if method not in metric_dimension.ALGORITHMS:
		return None

	M = metric_dimension.graph6_decode(graph)
	E = metric_dimension.edges(M)
	nV = M.shape[0]
	nE = E.shape[0]
	print('Graph: {}'.format(graph))
	print('Vertices: {}'.format(nV))
	print('Edges: {}'.format(nE))
	print()

	DV = metric_dimension.distance_matrix(M)
	DE = metric_dimension.edge_distance_matrix(E, DV)
	if edge:
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

	print('Metric dimension...')
	with timer_ms() as k_time:
		find = metric_dimension.create_find(P, method, limit)
		for k in range(nV, -1, -1):
			print('Trying k = {}... '.format(k), end='', flush=True)
			result, time = find.exact(k)
			print(timer_ms(time), end='')
			if result == metric_dimension.status.unknown:
				print(' [TIMEOUT]', end='')
			print()
			match result:
				case metric_dimension.status.unknown:
					k = -1
					break
				case metric_dimension.status.unsat:
					k = k + 1
					break
				case metric_dimension.status.sat:
					continue
		else:
			k = 0
	if edge:
		print('Edge metric dimension: {}'.format(k))
		print('Edge metric dimension time: {}'.format(k_time))
	else:
		print('Metric dimension: {}'.format(k))
		print('Metric dimension time: {}'.format(k_time))
	print()

option_spec = [
	(['-a', '--algorithm'], '', None),
	(['--edge'], False, parse_switch),
	(['--timeout'], 0, float)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	method, _, _ = cast(Tuple[str, bool, float], option)
	return method in metric_dimension.ALGORITHMS

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

			--timeout=<seconds>
				Stop search when over the specified amount of time.
				Defaults to 0, meaning no limit.
	'''
	).strip()
