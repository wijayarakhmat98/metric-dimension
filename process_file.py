#!/usr/bin/env python3.13

import copy
import graph_utils
import json
import metric_dimension
import multiprocessing
import re
import sys
from time import perf_counter

def file_to_list(filename):
	with open(filename, 'r') as file:
		ss = file.readlines()
		ss = [s.strip() for s in ss]
		return ss

def find_metric_dimension(s):
	start = perf_counter()
	m = graph_utils.graph6_decode(s)
	d = graph_utils.distance_matrix(m)
	e, _ = graph_utils.edge_distance_matrix(m, d)
	v = metric_dimension.create_node_boolean(m)
	b = metric_dimension.distance_similarity_broadcast(d)
	eb = metric_dimension.distance_similarity_broadcast(e)
	p = metric_dimension.distance_similarity_prune(b)
	ep = metric_dimension.distance_similarity_prune(eb)
	t = metric_dimension.distance_similarity_permute(p)
	et = metric_dimension.distance_similarity_permute(ep)
	u = metric_dimension.distance_similarity_group(t)
	eu = metric_dimension.distance_similarity_group(et)
	w_start = perf_counter()
	w = metric_dimension.find_least(v, u)
	w_end = perf_counter()
	n_start = perf_counter()
	n = metric_dimension.find_enumerate(v, u, len(w))
	n_end = perf_counter()
	nr = [metric_dimension.resolving_representation(v, w, d) for w in n]
	nc = [metric_dimension.valid(r) for r in nr]
	ns = [{'set': [str(x) for x in w], 'is_sane': c} for w, c in zip(n, nc)]
	ew_start = perf_counter()
	ew = metric_dimension.find_least(v, eu)
	ew_end = perf_counter()
	end = perf_counter()
	en_start = perf_counter()
	en = metric_dimension.find_enumerate(v, eu, len(ew))
	en_end = perf_counter()
	enr = [metric_dimension.resolving_representation(v, ew, e) for ew in en]
	enc = [metric_dimension.valid(er) for er in enr]
	ens = [{'set': [str(x) for x in ew], 'is_sane': ec} for ew, ec in zip(en, enc)]
	info = {
		'graph': s,
		'vertex': {
			'dimension': len(w),
			'time': w_end - w_start,
			'enumerate': {
				'n': len(n),
				'resolving': ns,
				'time': n_end - n_start
			}
		},
		'edge': {
			'dimension': len(ew),
			'time': ew_end - ew_start,
			'enumerate': {
				'n': len(en),
				'resolving': ens,
				'time': en_end - en_start
			}
		},
		'time': end - start
	}
	infos = copy.deepcopy(info)
	infos['vertex']['enumerate']['resolving'] = [str(s) for s in infos['vertex']['enumerate']['resolving']]
	infos['edge']['enumerate']['resolving'] = [str(s) for s in infos['edge']['enumerate']['resolving']]
	infos = json.dumps(infos, indent=2)
	infos = re.sub(r'"({\'set.*)"', r'\1', infos)
	return {'raw': info, 'str': infos}

def main(args):
	if len(args) != 2:
		print('usage:')
		print('\t<graph6 filename> <output_filename>')
		return

	graph6_filename = args[0]
	output_filename = args[1]

	ss = file_to_list(graph6_filename)

	with open(output_filename, 'w') as output_file:
		with multiprocessing.Pool() as pool:
			for result in pool.imap_unordered(find_metric_dimension, ss):
				print(result['str'])
				print(result['raw'], file=output_file)

if __name__ == '__main__':
	main(sys.argv[1:])
