import networkx as nx
import numba as nb # type: ignore
import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import floyd_warshall # type: ignore
from typing import cast, Tuple

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
	d = cast(npt.NDArray[np.float_], floyd_warshall(m, False)) # type: ignore
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
		for r, (j, k) in enumerate(es):
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
