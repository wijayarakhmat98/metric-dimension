import math
import metric_dimension
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
			with timer_ms() as k_time: found = find_exact(n, p, k)
			print(k_time)
			if not found:
				b = k + 1
				break
		else:
			b = 0
	print('Metric dimension: {}'.format(b))
	print('Metric dimension time: {}'.format(b_time))
	print()

def find_exact(n : int, p : npt.NDArray[np.bool_], k : int) -> bool:
	if k == 0:
		return False

	q0 = p[p.sum(axis=1) >= k]
	N = len(q0)

	q : List[npt.NDArray[np.bool_]] = []
	last : List[int] = []
	total_count = 0
	multiplier = 1

	q_ : List[npt.NDArray[np.bool_]] = []
	last_ : List[int] = []
	count = 0
	for j in range(N):
		row0 = q0[j]
		combine = row0
		c = math.comb(combine.sum(), k)
		if combine.sum() == 0:
			c = 0
		count += c
		is_carried = j < N - 1 and combine.sum() >= k
		if is_carried:
			q_.append(combine)
			last_.append(j)
	q = q_
	last = last_
	total_count += multiplier * count
	multiplier *= -1

	for _ in range(2, n):
		if len(q) == 0:
			break
		q_  = []
		last_  = []
		count = 0
		for i, row in enumerate(q):
			for j in range(last[i] + 1, N):
				row0 = q0[j]
				combine = row & row0
				c = math.comb(combine.sum(), k)
				if combine.sum() == 0:
					c = 0
				count += c
				is_carried = j < N - 1 and combine.sum() >= k
				if is_carried:
					q_.append(combine)
					last_.append(j)
		q = q_
		last = last_
		total_count += multiplier * count
		multiplier *= -1

	found = total_count < math.comb(n, k)
	return found
