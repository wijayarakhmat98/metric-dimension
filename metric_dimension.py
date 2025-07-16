import builtins
import networkx as nx
import numpy as np
import numpy.typing as npt
import numba as nb # type: ignore
import scipy as sp # type: ignore
from typing import Any, cast, List, Tuple
import z3 # type: ignore

def graph6_decode(s : str) -> Tuple[int, npt.NDArray[np.bool_]]:
	b = s.encode()
	g = nx.from_graph6_bytes(b) # type: ignore
	n = g.number_of_nodes()
	m = cast(npt.NDArray[np.bool_], nx.to_numpy_array(g, dtype=np.bool_)) # type: ignore
	return n, m

def graph6_encode(m : npt.NDArray[np.bool_]) -> str:
	g = cast(nx.Graph, nx.from_numpy_array(m)) # type: ignore
	b = cast(bytes, nx.to_graph6_bytes(g)) # type: ignore
	s = b.decode().strip()
	return s

def distance_matrix(m : npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
	d = cast(npt.NDArray[np.float_], sp.sparse.csgraph.floyd_warshall(m, False)) # type: ignore
	d = d.astype(np.int_)
	return d

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

__njit__distance_matrix_edge = nb.njit( # type: ignore
	fastmath=True,
	cache=True
)(__njit__distance_matrix_edge)

__njit__distance_matrix_edge.compile(( # type: ignore
	nb.types.intp,
	nb.types.Array(nb.types.boolean, 2, 'C'),
	nb.types.Array(nb.types.intp, 2, 'C'),
	nb.types.Array(nb.types.intp, 2, 'C')
))

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
	_vs = [z3.Int('x{}'.format(i + 1)) for i in range(n)] # type: ignore
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
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # type: ignore
	k = z3.Int('k') # type: ignore
	z.add(z3.Sum(*vs) == k) # type: ignore
	z.add(k >= 1) # type: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < k for c in p])) # type: ignore
	z.minimize(k) # type: ignore
	z.check() # type: ignore
	b = cast(int, z.model()[k].as_long()) # type: ignore
	return b

def enumerate(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_], n : int) -> npt.NDArray[np.bool_]:
	z = z3.Solver()
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # type: ignore
	z.add(z3.Sum(*vs) == n) # type: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < n for c in p])) # type: ignore
	ws : List[List[int]] = []
	while z.check() == z3.sat: # type: ignore
		model = z.model()
		w = [cast(int, model[v].as_long()) for v in vs] # type: ignore
		z.add(z3.Or([v != v_ for v, v_ in zip(vs, w)])) # type: ignore
		ws.append(w)
	return np.array(ws, dtype=np.bool_)

def resolving_representation(w : npt.NDArray[np.bool_], d : npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
	return d[:, w]

def is_resolving_valid(r : npt.NDArray[np.int_]) -> bool:
	r_ = np.unique(r, axis=0)
	valid = r.shape[0] == r_.shape[0]
	return valid
