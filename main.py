#!/usr/bin/env python3.12

import json
import multiprocessing
import metric_dimension
import numpy as np
import re
import subprocess
import sys
from typing import Any, Dict, List, Set, TextIO

def process(graph : str) -> str:
	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	es = metric_dimension.edges(m)

	d = metric_dimension.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)
	b = metric_dimension.find(vs, p)

	de = metric_dimension.distance_matrix_edge(n, d, es)
	pe = metric_dimension.distance_similarity(de)
	be = metric_dimension.find(vs, pe)

	return json.dumps({
		'graph': graph,
		'metric_dimension': b,
		'edge_dimension': be
	})

def resume(s : str) -> str:
	data : Dict[str, Any] = json.loads(s)
	graph : str = data['graph']
	return graph

def read_file(file : TextIO) -> List[str]:
	return [line.strip() for line in file]

def main_debug(graph : str, info : List[str]) -> None:
	n, m = metric_dimension.graph6_decode(graph)
	vs = metric_dimension.vertices(n)
	es = metric_dimension.edges(m)

	d = metric_dimension.distance_matrix(m)
	p = metric_dimension.distance_similarity(d)
	b = metric_dimension.find(vs, p)
	ws = metric_dimension.enumerate(vs, p, b)
	rs = np.stack([metric_dimension.resolving_representation(w, d) for w in ws])
	rs_valid = np.array([metric_dimension.is_resolving_valid(r) for r in rs]).reshape(-1, 1)

	de = metric_dimension.distance_matrix_edge(n, d, es)
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

def main_process(graphs_exclude : Set[str] = set()) -> None:
	graphs = set(read_file(sys.stdin)) - graphs_exclude
	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(process, graphs):
			print(result)

def main_resume(filename_output : str) -> None:
	with open(filename_output, 'r') as file:
		result = read_file(file)
	with multiprocessing.Pool() as pool:
		graphs_exclude = set(pool.imap_unordered(resume, result))
	main_process(graphs_exclude)

def main_usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				<debug|process|resume> ...

				debug <graph6 string> [info...]
				process
				resume <output file>

				Both process and resume read graphs from stdin.

			info:
				graph    : graph6 string

				n        : number of vertices
				m        : adjacency matrix
				vs       : vertices\' names
				es       : edges

				d        : distance matrix
				p        : distance similarity
				b        : metric dimension
				ws       : resolving sets
				rs       : resolving sets\' representations
				rs_valid : valid resolving sets?

				de       : edge distance matrix
				pe       : edge distance similarity
				be       : edge metric dimension
				wes      : edge resolving sets
				res      : edge resolving sets\' representations
				res_valid: valid edge resolving sets?
		'''
		).strip()
	)

def main(args : List[str]) -> None:
	if len(args) < 1 or args[0] in ('debug', 'resume') and len(args) < 2:
		main_usage()
		return
	match args[0]:
		case 'debug'  : main_debug(args[1], args[2:])
		case 'process': main_process()
		case 'resume' : main_resume(args[1])
		case _        : main_usage()

def run_mypy(filename: str) -> bool:
	result = subprocess.run(['mypy', filename], capture_output=True, text=True)
	if result.returncode == 0:
		return True
	print(result.stdout)
	print(result.stderr, file=sys.stderr)
	return False

def run_main() -> None:
	if not run_mypy(__file__):
		return
	if not run_mypy('metric_dimension.py'):
		return
	metric_dimension.__njit_compile__()
	main(sys.argv[1:])

if __name__ == '__main__':
	run_main()
