#!/usr/bin/env python3.13

import graph_utils
import metric_dimension
import numpy as np

def main():
	m = np.array([
		[0, 1, 1, 0, 0],
		[1, 0, 1, 1, 1],
		[1, 1, 0, 1, 0],
		[0, 1, 1, 0, 0],
		[0, 1, 0, 0, 0]
	])
	print(m, '\n')
	d = graph_utils.distance_matrix(m)
	print(d, '\n')
	v = metric_dimension.create_node_boolean(m)
	print(v, '\n')
	p = metric_dimension.distance_similarity_prune(d)
	print(p, '\n')
	t = metric_dimension.distance_similarity_permute(p)
	print(t, '\n')
	u = metric_dimension.distance_similarity_group(t)
	for i, c in enumerate(u):
		print(i)
		print(c, '\n')
	found, w = metric_dimension.find_least(v, u)
	print(found, w)
	if found:
		r = metric_dimension.resolving_representation(v, w, d)
		print(r)
		print(metric_dimension.valid(r))

if __name__ == '__main__':
	main()
