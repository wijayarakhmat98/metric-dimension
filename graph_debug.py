#!/usr/bin/env python3.12

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
	w = metric_dimension.find_least(v, p)
	if 'w' in info: print(w, '\n')
	if w != 0:
		n = metric_dimension.find_enumerate(v, p, w)
		if 'n' in info: print(n, '\n')
		nr = [metric_dimension.resolving_representation(v, w, d) for w in n]
		if 'nr' in info:
			for r in nr:
				print(r, '\n')
		nc = [metric_dimension.valid(r) for r in nr]
		if 'nc' in info: print(nc, '\n')
	ew = metric_dimension.find_least(v, ep)
	if 'ew' in info: print(ew, '\n')
	if ew != 0:
		en = metric_dimension.find_enumerate(v, ep, ew)
		if 'en' in info: print(en, '\n')
		enr = [metric_dimension.resolving_representation(v, ew, e) for ew in en]
		if 'enr' in info:
			for er in enr:
				print(er, '\n')
		enc = [metric_dimension.valid(er) for er in enr]
		if 'enc' in info: print(enc, '\n')

if __name__ == '__main__':
	main(sys.argv[1:])
