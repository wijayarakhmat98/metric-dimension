import builtins
import networkx as nx
import numba as nb # type: ignore
import numpy as np
import numpy.typing as npt
import scipy as sp # type: ignore
from typing import Any, cast, List, Tuple
import z3 # type: ignore

def graph6_decode(s : str) -> Tuple[int, npt.NDArray[np.bool_]]:
	b = s.encode()
	g = nx.from_graph6_bytes(b) # pyright: ignore
	n = g.number_of_nodes()
	m = cast(npt.NDArray[np.bool_], nx.to_numpy_array(g, dtype=np.bool_)) # type: ignore
	m = m.T
	return n, m

def graph6_encode(m : npt.NDArray[np.bool_]) -> str:
	m = m.T
	g = nx.from_numpy_array(m) # pyright: ignore
	b = cast(bytes, nx.to_graph6_bytes(g)) # pyright: ignore
	s = b.decode().strip()
	return s

def vertices(n : int) -> npt.NDArray[Any]:
	_vs = [z3.Int('x{}'.format(i + 1)) for i in range(n)] # pyright: ignore
	vs = np.array(_vs)
	return vs

def edges(m : npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
	m = np.triu(m, 1)
	es = np.argwhere(m)
	return es

def distance_matrix(m : npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
	m = m.T
	d = cast(npt.NDArray[np.float64], sp.sparse.csgraph.floyd_warshall(m, False))
	d = d.T
	d_ = d.astype(np.int_)
	return d_

def distance_matrix_edge(
	n : int,
	d : npt.NDArray[np.int_],
	es : npt.NDArray[np.intp]
) -> (
	npt.NDArray[np.int_]
):
	de = np.empty((len(es), n), dtype=np.int_)
	for i in range(n):
		for r, (j, k) in builtins.enumerate(es):
			de[r, i] = min(d[j, i], d[k, i])
	return de

def distance_similarity(d : npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
	N, n = d.shape
	p : npt.NDArray[np.bool_] = np.empty(((N - 1) * N // 2, n), dtype=np.bool_)
	k = 0
	for i, pivot in builtins.enumerate(d):
		for row in d[i + 1:]:
			similarity : npt.NDArray[np.bool_] = pivot == row
			if 0 < similarity.sum() < n:
				for j in range(k):
					if np.all(p[j] == similarity):
						break
				else:
					p[k] = similarity
					k += 1
	p = p[:k]
	p = p[np.argsort(p.sum(axis=1))]
	mask = np.ones(k, dtype=np.bool_)
	for i, pivot in builtins.enumerate(p):
		for row in p[i + 1:]:
			if np.all((pivot | row) == row):
				mask[i] = np.False_
				break
	p = p[mask]
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

def enumerate(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_], k : int) -> npt.NDArray[np.bool_]:
	z = z3.Solver()
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # pyright: ignore
	z.add(z3.Sum(*vs) == k) # pyright: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < k for c in p])) # pyright: ignore
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

distance_matrix_edge = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(distance_matrix_edge)

distance_similarity = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(distance_similarity)

def __njit_compile__() -> None:
	distance_matrix_edge.compile( # type: ignore
		(nb.types.intp, nb.types.Array(nb.types.intp, 2, 'C'), nb.types.Array(nb.types.intp, 2, 'C'))
	)
	distance_similarity.compile( # type: ignore
		(nb.types.Array(nb.types.intp, 2, 'C'),)
	)
