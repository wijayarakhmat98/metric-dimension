import math
import metric_dimension
import numpy as np
import numpy.typing as npt
import re
import sys
from typing import cast, List, Tuple
from utils import parse_args, parse_switch, timer

class timer_ms(timer):
	def __str__(self) -> str:
		return '{}ms'.format(round(1000 * float(self), 3))

def debug(graph : str, option_raw : List[str], *args : object, **kwargs : object) -> None:
	help, k, l, print_p, print_q = cast(
		Tuple[bool, int, int, bool, bool],
		parse_args(option_raw, [
			(['-h', '--help'], False, parse_switch),
			(['-k'], 1, int),
			(['-l'], -1, int),
			(['-p'], False, parse_switch),
			(['-q'], False, parse_switch)
		])
	)

	if help:
		usage()

	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	print('Graph: {}'.format(graph))
	print('Vertices: {}'.format(len(vs)))
	print()

	d = metric_dimension.distance_matrix(m)

	p = metric_dimension.distance_similarity(d)
	if print_p:
		print('Distance similarity...')
		for row in p:
			print('  {}'.format(row.astype(int)))
		print()

	q0 = p[p.sum(axis=1) >= k]
	N = len(q0)

	q : List[npt.NDArray[np.bool_]] = []
	last : List[int] = []
	total_count = 0
	multiplier = 1

	print('Choose 1 constraints...')
	q_ : List[npt.NDArray[np.bool_]] = []
	last_ : List[int] = []
	count = 0
	number_of_constraints = 0
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
		if print_q:
			print('  {} [{}] -> {} ({})'.format(combine.astype(int), j, c, 'Carried' if is_carried else 'Removed'))
		number_of_constraints += 1
	q = q_
	last = last_
	total_count += multiplier * count
	multiplier *= -1
	print('Number of constraints: {}'.format(number_of_constraints))
	print('Count: {}'.format(count))
	print('Total count: {}'.format(total_count))
	print()

	for stage in range(2, n):
		if stage == l + 1:
			break
		if len(q) == 0:
			break
		print('Choose {} constraints...'.format(stage))
		q_  = []
		last_  = []
		count = 0
		number_of_constraints = 0
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
				if print_q:
					print('  {} [{}] -> {} ({})'.format(combine.astype(int), j, c, 'Carried' if is_carried else 'Removed'))
				number_of_constraints += 1
		q = q_
		last = last_
		total_count += multiplier * count
		multiplier *= -1
		print('Number of constraints: {}'.format(number_of_constraints))
		print('Count: {}'.format(count))
		print('Total count: {}'.format(total_count))
		print()

def usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				... <-k=<cadinality>> [-l=<limit>] [-p0] [-p]

			options:
				-k=<cardinality>
					Count number of invalid solutions for resolving set with cardinality k.
					Defaults to 1.

				-l=<limit>
					Limit the combination iteration.

				-p
					Print the original distance similarity matrix.

				-q
					Print the resulting distance similarity matrix at each step.
		'''
		).strip(),
		file=sys.stderr
	)
	sys.exit()
