#!/usr/bin/env python3.12

import numpy as np
import metric_dimension
import sys
from typing import List

def main(args : List[str]) -> None:
	if len(args) == 0:
		print('usage:')
		print('\t<graph6 string> [info...]')
		print()
		print('info:')
		print('\tgraph    : graph6 string')
		print()
		print('\tn        : number of vertices')
		print('\tm        : adjacency matrix')
		print('\tvs       : vertices\' names')
		print('\tes       : edges')
		print()
		print('\td        : distance matrix')
		print('\tp        : distance similarity')
		print('\tb        : metric dimension')
		print('\tws       : resolving sets')
		print('\trs       : resolving sets\' representations')
		print('\trs_valid : valid resolving sets?')
		print()
		print('\tde       : edge distance matrix')
		print('\tpe       : edge distance similarity')
		print('\tbe       : edge metric dimension')
		print('\twes      : edge resolving sets')
		print('\tres      : edge resolving sets\' representations')
		print('\tres_valid: valid edge resolving sets?')
		return

	graph = args[0]
	info = args[1:]

	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.variable(n)

	d = metric_dimension.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)
	b = metric_dimension.find(vs, p)
	ws = metric_dimension.enumerate(vs, p, b)
	rs = np.stack([metric_dimension.resolving_representation(w, d) for w in ws])
	rs_valid = np.array([metric_dimension.is_resolving_valid(r) for r in rs]).reshape(-1, 1)

	de, es = metric_dimension.distance_matrix_edge(n, m, d)
	pe = metric_dimension.distance_similarity(de)
	be = metric_dimension.find(vs, pe)
	wes = metric_dimension.enumerate(vs, pe, be)
	res = np.stack([metric_dimension.resolving_representation(we, de) for we in wes])
	res_valid = np.array([metric_dimension.is_resolving_valid(re) for re in res]).reshape(-1, 1)

	graph = metric_dimension.graph6_encode(m)
	es = np.array([[vs[e[0]], vs[e[1]]] for e in es])
	ws = np.stack([vs[w] for w in ws])
	wes = np.stack([vs[we] for we in wes])

	log : List[str] = []
	for i in info:
		match i:
			case 'graph'    : log.append(graph)
			case 'n'        : log.append(str(n))
			case 'm'        : log.append(str(m))
			case 'vs'       : log.append(str(vs))
			case 'es'       : log.append(str(es))
			case 'd'        : log.append(str(d))
			case 'p'        : log.append(str(p))
			case 'b'        : log.append(str(b))
			case 'ws'       : log.append(str(ws))
			case 'rs'       : log.append(str(rs))
			case 'rs_valid' : log.append(str(rs_valid))
			case 'de'       : log.append(str(de))
			case 'pe'       : log.append(str(pe))
			case 'be'       : log.append(str(be))
			case 'wes'      : log.append(str(wes))
			case 'res'      : log.append(str(res))
			case 'res_valid': log.append(str(res_valid))
			case _          : pass
	print('\n\n'.join(log))

if __name__ == '__main__':
	main(sys.argv[1:])
