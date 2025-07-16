#!/usr/bin/env python3.12

import builtins
import json
import multiprocessing
import networkx as nx
import numpy as np
import numpy.typing as npt
import numba as nb # type: ignore
import re
import scipy as sp # type: ignore
import subprocess
import sys
from typing import Any, cast, List, TextIO, Tuple
import z3 # type: ignore

def read_file(file : TextIO) -> List[str]:
	return [line.strip() for line in file]

def graph6_decode(s : str) -> Tuple[int, npt.NDArray[np.bool_]]:
	b = s.encode()
	g = nx.from_graph6_bytes(b) # pyright: ignore
	n = g.number_of_nodes()
	m = cast(npt.NDArray[np.bool_], nx.to_numpy_array(g, dtype=np.bool_)) # type: ignore
	m = m.T
	return n, m

def graph6_encode(m : npt.NDArray[np.bool_]) -> str:
	m = m.T
	g = cast(nx.Graph, nx.from_numpy_array(m)) # type: ignore
	b = cast(bytes, nx.to_graph6_bytes(g)) # pyright: ignore
	s = b.decode().strip()
	return s

def distance_matrix(m : npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
	m = m.T
	d = cast(npt.NDArray[np.float_], sp.sparse.csgraph.floyd_warshall(m, False))
	d = d.T
	d_ = d.astype(np.int_)
	return d_

def __njit__distance_matrix_edge(
	n : int,
	m : npt.NDArray[np.bool_],
	d : npt.NDArray[np.int_],
	es : npt.NDArray[np.intp]
) -> (
	npt.NDArray[np.int_]
):
	de = np.empty((m.sum(), n), dtype=np.int_)
	for i in range(n):
		for r, (j, k) in builtins.enumerate(es):
			de[r, i] = min(d[j, i], d[k, i])
	return de

def distance_matrix_edge(
	n : int,
	m : npt.NDArray[np.bool_],
	d : npt.NDArray[np.int_]
) -> Tuple[
	npt.NDArray[np.intp],
	npt.NDArray[np.int_]
]:
	m = m.copy()
	m[np.tril_indices(n)] = np.False_
	es = np.argwhere(m)
	de = __njit__distance_matrix_edge(n, m, d, es)
	return de, es

def variable(n : int) -> npt.NDArray[Any]:
	_vs = [z3.Int('x{}'.format(i + 1)) for i in range(n)] # pyright: ignore
	vs = np.array(_vs)
	return vs

def distance_similarity(d : npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
	p : npt.NDArray[np.bool_] = d[None, :, :] == d[:, None, :]
	p = p.reshape(-1, p.shape[2])
	p = np.unique(p, axis=0)
	p = p[np.any(p, axis=1) & ~np.all(p, axis=1)]
	return p

def find(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_]) -> int:
	z = z3.Optimize()
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # pyright: ignore
	k = z3.Int('k') # pyright: ignore
	z.add(z3.Sum(*vs) == k) # pyright: ignore
	z.add(k >= 1) # pyright: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < k for c in p])) # pyright: ignore
	z.minimize(k) # pyright: ignore
	z.check() # pyright: ignore
	b = cast(int, z.model()[k].as_long()) # pyright: ignore
	return b

def enumerate(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_], n : int) -> npt.NDArray[np.bool_]:
	z = z3.Solver()
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # pyright: ignore
	z.add(z3.Sum(*vs) == n) # pyright: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < n for c in p])) # pyright: ignore
	ws : List[List[int]] = []
	while z.check() == z3.sat: # pyright: ignore
		model = z.model()
		w = [cast(int, model[v].as_long()) for v in vs] # pyright: ignore
		z.add(z3.Or([v != v_ for v, v_ in zip(vs, w)])) # pyright: ignore
		ws.append(w)
	ws_ = np.array(ws, dtype=np.bool_)
	return ws_

def resolving_representation(w : npt.NDArray[np.bool_], d : npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
	return d[:, w]

def is_resolving_valid(r : npt.NDArray[np.int_]) -> bool:
	r_ = np.unique(r, axis=0)
	valid = r.shape[0] == r_.shape[0]
	return valid

def main_debug(graph : str, info : List[str]) -> None:
	n, m = graph6_decode(graph)
	vs = variable(n)

	d = distance_matrix(m)
	p = distance_similarity(d)
	b = find(vs, p)
	ws = enumerate(vs, p, b)
	rs = np.stack([resolving_representation(w, d) for w in ws])
	rs_valid = np.array([is_resolving_valid(r) for r in rs]).reshape(-1, 1)

	de, es = distance_matrix_edge(n, m, d)
	pe = distance_similarity(de)
	be = find(vs, pe)
	wes = enumerate(vs, pe, be)
	res = np.stack([resolving_representation(we, de) for we in wes])
	res_valid = np.array([is_resolving_valid(re) for re in res]).reshape(-1, 1)

	graph = graph6_encode(m)
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

def process(graph : str) -> str:
	n, m = graph6_decode(graph)
	vs = variable(n)

	d = distance_matrix(m)
	p = distance_similarity(d)
	b = find(vs, p)

	de, _ = distance_matrix_edge(n, m, d)
	pe = distance_similarity(de)
	be = find(vs, pe)

	return json.dumps({
		'graph': graph,
		'metric_dimension': b,
		'edge_dimension': be
	})

def main_process() -> None:
	graphs = read_file(sys.stdin)
	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(process, graphs):
			print(result)

def print_usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				<debug|process> ...

				debug <graph6 string> [info...]
				process [info...]

				The subcommand process read graphs from stdin.

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
	if len(args) < 1 or args[0] in ('debug') and len(args) < 2:
		print_usage()
		return
	match args[0]:
		case 'debug'  : main_debug(args[1], args[2:])
		case 'process': main_process()
		case _        : print_usage()

def run_mypy(filename: str) -> bool:
	result = subprocess.run(['mypy', filename], capture_output=True, text=True)
	if result.returncode == 0:
		return True
	print(result.stdout)
	print(result.stderr, file=sys.stderr)
	return False

__njit__distance_matrix_edge = nb.njit( # pyright: ignore
	fastmath=True,
	cache=True
)(__njit__distance_matrix_edge)

def run_main() -> None:
	if not run_mypy(__file__):
		return
	__njit__distance_matrix_edge.compile(( # type: ignore
		nb.types.intp,
		nb.types.Array(nb.types.boolean, 2, 'C'),
		nb.types.Array(nb.types.intp, 2, 'C'),
		nb.types.Array(nb.types.intp, 2, 'C')
	))
	main(sys.argv[1:])

if __name__ == '__main__':
	run_main()
