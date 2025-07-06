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
	w = metric_dimension.find_least(v, u)
	if 'w' in info: print(w, '\n')
	if len(w) != 0:
		r = metric_dimension.resolving_representation(v, w, d)
		if 'r' in info: print(r, '\n')
		if 'c' in info: print(metric_dimension.valid(r), '\n')
		n = metric_dimension.find_enumerate(v, u, len(w))
		if 'n' in info: print(n, '\n')
		nr = [metric_dimension.resolving_representation(v, w, d) for w in n]
		if 'nr' in info:
			for r in nr:
				print(r, '\n')
		nc = [metric_dimension.valid(r) for r in nr]
		if 'nc' in info: print(nc, '\n')
	ew = metric_dimension.find_least(v, eu)
	if 'ew' in info: print(ew, '\n')
	if len(ew) != 0:
		er = metric_dimension.resolving_representation(v, ew, e)
		if 'er' in info: print(er, '\n')
		if 'ec' in info: print(metric_dimension.valid(er), '\n')
		en = metric_dimension.find_enumerate(v, eu, len(ew))
		if 'en' in info: print(en, '\n')
		enr = [metric_dimension.resolving_representation(v, ew, e) for ew in en]
		if 'enr' in info:
			for er in enr:
				print(er, '\n')
		enc = [metric_dimension.valid(r) for er in enr]
		if 'enc' in info: print(enc, '\n')

if __name__ == '__main__':
	main(sys.argv[1:])
