#!/usr/bin/env python3.13

import graph_utils
import metric_dimension
import numpy as np
import z3

def find_least(v, constraint):
	count = z3.Sum(*[z3.If(b, 0, 1) for b in v])
	o = z3.Optimize()
	o.add(constraint)
	o.minimize(count)
	if o.check() != z3.sat:
		return z3.unsat, -1, []
	model = o.model()
	w = [b for b in v if z3.is_false(model.evaluate(b))]
	return z3.sat, len(w), w

def get_resolving_representation(v, d, w):
	mask = np.array([b in w for b in v])
	return d[:, mask]

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
	sat, n, w = find_least(v, c)
	z = get_resolving_representation(v, d, w)
	print(m, v, d, p, c, sat, n, w, z, sep='\n\n')

if __name__ == '__main__':
	main()
