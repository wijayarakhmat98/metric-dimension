import itertools
import metric_dimension
import numpy as np
import numpy.typing as npt
from typing import Any, Tuple
from utils import timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option : Tuple[Any, ...]) -> None:
	M = metric_dimension.graph6_decode(graph)
	nV = M.shape[0]
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

	print('Metric dimension...')
	with timer_ms() as total_time:
		for k in range(nV - 1, -1, -1):
			print('Trying k = {}... '.format(k), end='', flush=True)
			with timer_ms() as k_time:
				found = find_exact(P, k)
			print(k_time)
			if not found:
				k = k + 1
				break
		else:
			k = 0
	print('Metric dimension: {}'.format(k))
	print('Metric dimension time: {}'.format(total_time))
	print()

option_spec = None
option_valid = None

def help() -> str:
	return ''

def find_exact(P : npt.NDArray[np.bool_], k : int) -> bool:
	nV = P.shape[0]
	_P = P[:, P.sum(axis=0) >= k]
	combinations = itertools.combinations(range(nV), k)
	for indices in combinations:
		W = np.zeros((nV, 1), dtype=np.bool_)
		W[indices, 0] = True
		is_subset = np.all((W | _P) == _P, axis=0)
		if not np.any(is_subset):
			return True
	return False
