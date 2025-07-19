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

def find_constraint(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_]) -> Tuple[List[z3.BoolRef], z3.ArithRef]:
	ck = z3.Int('k') # pyright: ignore
	cs : List[z3.BoolRef] = []
	for v in vs:
		cs.append(v >= 0)
		cs.append(v <= 1)
	cs.append(z3.Sum(*vs) == ck) # pyright: ignore
	for vf in p:
		cs.append(z3.Sum(*vs[vf]) < ck) # pyright: ignore
	return cs, ck

def find_exact(cs : List[z3.BoolRef], ck : z3.ArithRef, k : int) -> bool:
	z = z3.Solver()
	z.add(cs) # pyright: ignore
	z.add(ck == k) # pyright: ignore
	found = cast(bool, z.check() == z3.sat) # pyright: ignore
	return found

def find(n : int, vs : npt.NDArray[Any], p : npt.NDArray[np.bool_]) -> int:
	cs, ck = find_constraint(vs, p)
	for k in range(1, n):
		if find_exact(cs, ck, k):
			return k
	return 0

def find_exact_bruteforce(n : int, p : npt.NDArray[np.bool_], k : int) -> bool:
	p = p[p.sum(axis=1) >= k]
	idx = np.array(list(range(k)), dtype=np.int_)
	idx_last = (n - idx - 1)[::-1]
	choice = np.zeros(n, dtype=np.bool_)
	choice[idx] = np.True_
	while True:
		for row in p:
			if np.all((choice | row) == row):
				break
		else:
			return True
		for i in range(k - 1, -1, -1):
			if idx[i] != idx_last[i]:
				break
		else:
			return False
		choice[idx[i:]] = np.False_
		idx[i] += 1
		for j in range(i + 1, k):
			idx[j] = idx[j - 1] + 1
		choice[idx[i:]] = np.True_

def find_bruteforce(n : int, p : npt.NDArray[np.bool_]) -> int:
	for k in range(1, n):
		if find_exact_bruteforce(n, p, k):
			return k
	return 0

def enumerate(n : int, p : npt.NDArray[np.bool_], r : int) -> npt.NDArray[np.bool_]:
	c = 1
	for i in range(r):
		c *= n - i
		c //= i + 1
	ws : npt.NDArray[np.bool_] = np.empty((c, n), dtype=np.bool_)
	k = 0
	idx = np.array(list(range(r)), dtype=np.int_)
	idx_last = (n - idx - 1)[::-1]
	choice = np.zeros(n, dtype=np.bool_)
	choice[idx] = np.True_
	while True:
		for row in p:
			if np.all((choice | row) == row):
				break
		else:
			ws[k] = choice
			k += 1
		for i in range(r - 1, -1, -1):
			if idx[i] != idx_last[i]:
				break
		else:
			ws = ws[:k]
			return ws
		choice[idx[i:]] = np.False_
		idx[i] += 1
		for j in range(i + 1, r):
			idx[j] = idx[j - 1] + 1
		choice[idx[i:]] = np.True_

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

find_exact_bruteforce = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(find_exact_bruteforce)

find_bruteforce = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(find_bruteforce)

enumerate = nb.njit( # pyright: ignore
	fastmath=True, cache=True
)(enumerate)

def __njit_compile__() -> None:
	distance_matrix_edge.compile( # type: ignore
		(nb.types.intp, nb.types.Array(nb.types.intp, 2, 'C'), nb.types.Array(nb.types.intp, 2, 'C'))
	)
	distance_similarity.compile( # type: ignore
		(nb.types.Array(nb.types.intp, 2, 'C'),)
	)
	find_exact_bruteforce.compile( # type: ignore
		(nb.types.intp, nb.types.Array(nb.types.bool, 2, 'C'), nb.types.intp)
	)
	find_bruteforce.compile( # type: ignore
		(nb.types.intp, nb.types.Array(nb.types.bool, 2, 'C'))
	)
	enumerate.compile( # type: ignore
		(nb.types.intp, nb.types.Array(nb.types.bool, 2, 'C'), nb.types.intp)
	)
