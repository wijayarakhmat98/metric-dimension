#!/usr/bin/env python3.12

import graph_utils
import json
import metric_dimension
import multiprocessing
import sys
from typing import List
import utils

def process(graph : str) -> str:
	n, m = graph_utils.graph6_decode(graph)
	vs = metric_dimension.variable(n)

	d = graph_utils.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)
	b = metric_dimension.find(vs, p)

	de, _ = graph_utils.distance_matrix_edge(n, m, d)
	pe = metric_dimension.distance_similarity(de)
	be = metric_dimension.find(vs, pe)

	return json.dumps({
		'graph': graph,
		'metric_dimension': b,
		'edge_dimension': be
	})

def main(args : List[str]) -> None:
	if len(args) != 1:
		print('usage:')
		print('\t<graph6 filename>')
		return

	graph6_filename = args[0]

	graphs = utils.file_to_list(graph6_filename)

	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(process, graphs):
			print(result)

if __name__ == '__main__':
	main(sys.argv[1:])
