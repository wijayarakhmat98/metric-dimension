#!/usr/bin/env python3.12

import graph_utils
import json
import metric_dimension
import multiprocessing
import sys
import utils

def find_metric_dimension(s):
	m = graph_utils.graph6_decode(s)
	v = metric_dimension.create_node_boolean(m)

	d = graph_utils.distance_matrix(m)
	b = metric_dimension.distance_similarity_broadcast(d)
	p = metric_dimension.distance_similarity_prune(b)

	e, _ = graph_utils.edge_distance_matrix(m, d)
	eb = metric_dimension.distance_similarity_broadcast(e)
	ep = metric_dimension.distance_similarity_prune(eb)

	w = metric_dimension.find_least(v, p)
	ew = metric_dimension.find_least(v, ep)

	return json.dumps({
		'graph': s,
		'metric_dimension': w,
		'edge_dimension': ew
	})

def main(args):
	if len(args) != 1:
		print('usage:')
		print('\t<graph6 filename>')
		return

	graph6_filename = args[0]

	ss = utils.file_to_list(graph6_filename)

	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(find_metric_dimension, ss):
			print(result)

if __name__ == '__main__':
	main(sys.argv[1:])
