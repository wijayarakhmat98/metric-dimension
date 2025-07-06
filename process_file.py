#!/usr/bin/env python3.13

import graph_utils
import metric_dimension
import multiprocessing
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
	ew_start = perf_counter()
	ew = metric_dimension.find_least(v, eu)
	ew_end = perf_counter()
	end = perf_counter()
	return {
		'graph': s,
		'vertex': {
			'dimension': len(w),
			'time': w_end - w_start
		},
		'edge': {
			'dimension': len(ew),
			'time': ew_end - ew_start
		},
		'time': end - start
	}

def main(args):
	if len(args) != 1:
		print('usage:')
		print('\t<graph6 filename>')
		return

	filename = args[0]
	ss = file_to_list(filename)

	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(find_metric_dimension, ss):
			print(result)

if __name__ == '__main__':
	main(sys.argv[1:])
