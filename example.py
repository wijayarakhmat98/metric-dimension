#!/usr/bin/env python3.13

import graph_utils
import metric_dimension
import numpy as np
import z3

def main():
	m = np.array([
		[0, 1, 1, 0, 0],
		[1, 0, 1, 1, 1],
		[1, 1, 0, 1, 0],
		[0, 1, 1, 0, 0],
		[0, 1, 0, 0, 0]
	])
	v = metric_dimension.create_node_boolean(m)
	d = graph_utils.distance_matrix(m)
	p = metric_dimension.distance_similarity_prune(d)
	r = metric_dimension.apply_boolean_similarity(v, p)
	c = z3.Not(z3.Or(*r))
	sat, n, w = metric_dimension.find_least(v, c)
	z = metric_dimension.get_resolving_representation(v, d, w)
	print(m, v, d, p, c, sat, n, w, z, sep='\n\n')

if __name__ == '__main__':
	main()
