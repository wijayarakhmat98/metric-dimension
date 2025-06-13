#!/usr/bin/env python3.13

import graph_utils
import metric_dimension
import numpy as np
import sys

def main(args):
	if len(args) != 1 and len(args) != 2:
		print('usage:')
		print('\t<graph6 string> [info...]')
		print()
		print('info:')
		print('\tm\tadjacency matrix')
		print('\td\tdistance matrix')
		print('\tv\tnodes')
		print('\tb\tdistance similarity (broadcast)')
		print('\tp\tdistance similarity (prune)')
		print('\tt\tdistance similarity (permute)')
		print('\tu\tdistance similarity (group)')
		print('\tf\tsatisfiable')
		print('\tw\tresolving set')
		print('\tr\tresolving representation')
		print('\tc\tvalid solution? (sanity check)')
		return
	s = args[0]
	info = list(args[1] if len(args) == 2 else "")
	m = graph_utils.graph6_decode(s)
	if 'm' in info: print(m, '\n')
	d = graph_utils.distance_matrix(m)
	if 'd' in info: print(d, '\n')
	v = metric_dimension.create_node_boolean(m)
	if 'v' in info: print(v, '\n')
	b = metric_dimension.distance_similarity_broadcast(d)
	if 'b' in info: print(b, '\n')
	p = metric_dimension.distance_similarity_prune(b)
	if 'p' in info: print(p, '\n')
	t = metric_dimension.distance_similarity_permute(p)
	if 't' in info: print(t, '\n')
	u = metric_dimension.distance_similarity_group(t)
	if 'u' in info:
		for i, c in enumerate(u):
			print(i)
			print(c, '\n')
	found, w = metric_dimension.find_least(v, u)
	if 'f' in info: print(found, '\n')
	if 'w' in info: print(w, '\n')
	if found:
		r = metric_dimension.resolving_representation(v, w, d)
		if 'r' in info: print(r, '\n')
		if 'c' in info: print(metric_dimension.valid(r), '\n')

if __name__ == '__main__':
	main(sys.argv[1:])
