#!/usr/bin/env python3.13

import graph_utils
import metric_dimension
import sys

def main(args):
	if len(args) == 0:
		print('usage:')
		print('\t<graph6 string> [info...]')
		print()
		print('info:')
		print('\tm\tadjacency matrix')
		print('\tg\tgraph6 (sanity check)')
		print('\td\tdistance matrix')
		print('\te\tedge distance matrix')
		print('\tv\tnodes')
		print('\tb\tdistance similarity (broadcast)')
		print('\teb\tedge distance similarity (broadcast)')
		print('\tp\tdistance similarity (prune)')
		print('\tep\tedge distance similarity (prune)')
		print('\tt\tdistance similarity (permute)')
		print('\tet\tedge distance similarity (permute)')
		print('\tu\tdistance similarity (group)')
		print('\teu\tedge distance similarity (group)')
		print('\tf\tsatisfiable')
		print('\tef\tedge satisfiable')
		print('\tw\tresolving set')
		print('\tew\tedge resolving set')
		print('\tr\tresolving representation')
		print('\ter\tedge resolving representation')
		print('\tc\tvalid solution? (sanity check)')
		print('\tec\tedge valid solution? (sanity check)')
		return
	s = args[0]
	info = args[1:]
	m = graph_utils.graph6_decode(s)
	if 'm' in info: print(m, '\n')
	g = graph_utils.graph6_encode(m)
	if 'g' in info: print(g, '\n')
	d = graph_utils.distance_matrix(m)
	if 'd' in info: print(d, '\n')
	e, l = graph_utils.edge_distance_matrix(m, d)
	if 'e' in info: print(e, '\n')
	if 'l' in info: print(l, '\n')
	v = metric_dimension.create_node_boolean(m)
	if 'v' in info: print(v, '\n')
	b = metric_dimension.distance_similarity_broadcast(d)
	if 'b' in info: print(b, '\n')
	eb = metric_dimension.distance_similarity_broadcast(e)
	if 'eb' in info: print(eb, '\n')
	p = metric_dimension.distance_similarity_prune(b)
	if 'p' in info: print(p, '\n')
	ep = metric_dimension.distance_similarity_prune(eb)
	if 'ep' in info: print(ep, '\n')
	t = metric_dimension.distance_similarity_permute(p)
	if 't' in info: print(t, '\n')
	et = metric_dimension.distance_similarity_permute(ep)
	if 'et' in info: print(et, '\n')
	u = metric_dimension.distance_similarity_group(t)
	if 'u' in info:
		for i, c in enumerate(u):
			print(i)
			print(c, '\n')
	eu = metric_dimension.distance_similarity_group(et)
	if 'eu' in info:
		for i, c in enumerate(eu):
			print(i)
			print(c, '\n')
	found, w = metric_dimension.find_least(v, u)
	if 'f' in info: print(found, '\n')
	if 'w' in info: print(w, '\n')
	if found:
		r = metric_dimension.resolving_representation(v, w, d)
		if 'r' in info: print(r, '\n')
		if 'c' in info: print(metric_dimension.valid(r), '\n')
	efound, ew = metric_dimension.find_least(v, eu)
	if 'ef' in info: print(efound, '\n')
	if 'ew' in info: print(ew, '\n')
	if efound:
		er = metric_dimension.resolving_representation(v, ew, e)
		if 'er' in info: print(er, '\n')
		if 'ec' in info: print(metric_dimension.valid(er), '\n')

if __name__ == '__main__':
	main(sys.argv[1:])
