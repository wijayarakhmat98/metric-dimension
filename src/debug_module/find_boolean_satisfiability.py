import metric_dimension
import numpy as np
from typing import Any, Tuple
from utils import timer
import z3 # type: ignore

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option : Tuple[Any, ...]) -> None:
	M = metric_dimension.graph6_decode(graph)
	V = metric_dimension.vertices(M)
	nV = len(V)
	print('Graph: {}'.format(graph))
	print('Vertices: {}'.format(nV))
	print()

	DV = metric_dimension.distance_matrix(M)
	D = DV

	print('Distance similarity...')
	with timer_ms() as B_time: B = metric_dimension.distance_similarity(D)
	print('Distance similarity time: {}'.format(B_time))
	print()

	print('Reduced distance similarity...')
	with timer_ms() as P_time: P = metric_dimension.reduced_distance_similarity(B)
	print('Reduced distance similarity time: {}'.format(P_time))
	print()

	X = np.array([z3.Bool('x{}'.format(v + 1)) for v in V]) # pyright: ignore
	COL = P.shape[1]
	s = z3.Solver()
	s.add(z3.Or(*X)) # pyright: ignore
	for i in range(COL):
		s.add(z3.Implies(z3.Or(*X[P[:, i]]), z3.Or(*X[~P[:, i]]))) # pyright: ignore

	print('Metric dimension...')
	with timer_ms() as k_time:
		for k in range(nV - 1, -1, -1):
			print('Trying k = {}... '.format(k), end='', flush=True)
			s.push()
			s.add(z3.AtLeast(*X, k)) # pyright: ignore
			s.add(z3.AtMost(*X, k)) # pyright: ignore
			with timer_ms() as k_time:
				found : bool = s.check() == z3.sat # pyright: ignore
			print(k_time)
			if not found:
				k = k + 1
				break
			s.pop()
		else:
			k = -1
	print('Metric dimension: {}'.format(k))
	print('Metric dimension time: {}'.format(k_time))
	print()

option_spec = None
option_valid = None

def help() -> str:
	return ''
