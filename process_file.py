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
	v = metric_dimension.create_node_boolean(m)
	b = metric_dimension.distance_similarity_broadcast(d)
	p = metric_dimension.distance_similarity_prune(b)
	t = metric_dimension.distance_similarity_permute(p)
	u = metric_dimension.distance_similarity_group(t)
	_, w = metric_dimension.find_least(v, u)
	end = perf_counter()
	return (s, len(w), end - start)

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
