import metric_dimension
import numba as nb # type: ignore
import numpy as np
import numpy.typing as npt
from typing import List
from utils import timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, *args : object, **kwargs : object) -> None:
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
			with timer_ms() as k_time: found = find_exact(n, p[p.sum(axis=1) >= k], k)
			print(k_time)
			if not found:
				b = k + 1
				break
		else:
			b = 0
	print('Metric dimension: {}'.format(b))
	print('Metric dimension time: {}'.format(b_time))
	print()

def combination(n : int, r : int) -> int:
	c = 1
	for i in range(r):
		c *= n - i
		c //= i + 1
	return c

def find_exact(n : int, q0 : npt.NDArray[np.bool], k : int) -> bool:
	if k == 0:
		return False

	N = len(q0)

	q : List[npt.NDArray[np.bool_]] = []
	last : List[int] = []
	count = 0
	for j in range(N):
		row0 = q0[j]
		if row0.sum() == 0:
			c = 0
		else:
			c = combination(row0.sum(), k)
		count += c
		if j < N - 1 and row0.sum() >= k:
			q.append(row0)
			last.append(j)

	multiplier = -1

	for _ in range(2, n):
		if len(q) == 0:
			break

		q_ : List[npt.NDArray[np.bool_]] = []
		last_ : List[int] = []
		for i, row in enumerate(q):
			for j in range(last[i] + 1, N):
				row0 = q0[j]
				combine = row & row0
				if combine.sum() == 0:
					c = 0
				else:
					c = combination(combine.sum(), k)
				count += multiplier * c
				if j < N - 1 and combine.sum() >= k:
					q_.append(combine)
					last_.append(j)

		q = q_
		last = last_
		multiplier *= -1

	found = count < combination(n, k)
	return found

combination = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(combination)

combination.compile( # type: ignore
	(nb.types.intp, nb.types.intp)
)

find_exact = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(find_exact)

find_exact.compile( # type: ignore
	(nb.types.intp, nb.types.Array(nb.types.intp, 2, 'C'), nb.types.intp)
)
