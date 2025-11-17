import itertools
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy as sp # type: ignore
from typing import Any, cast, Tuple
import z3 # type: ignore

def graph6_decode(s : str) -> npt.NDArray[np.float64]:
	b = s.encode()
	G = nx.from_graph6_bytes(b) # pyright: ignore
	M = nx.to_numpy_array(G).astype(np.float64)
	return M

def graph6_encode(M : npt.NDArray[np.float64]) -> str:
	G = nx.from_numpy_array(M)
	b = cast(bytes, nx.to_graph6_bytes(G)) # pyright: ignore
	s = b.decode().strip()
	return s

def vertices(M : npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
	V = cast(npt.NDArray[np.int_], np.arange(M.shape[0]))
	return V

def edges(M : npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
	E = np.argwhere(np.triu(M, 1))
	return E

def distance_matrix(M : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	DV = cast(npt.NDArray[np.float64], sp.sparse.csgraph.floyd_warshall(M, False))
	return DV

def edge_distance_matrix(E : npt.NDArray[np.int_], DV : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	nV = DV.shape[0]
	nE = len(E)
	DE = np.empty((nV, nE), dtype=np.float64)
	for i in range(nV):
		for j, (k, l) in enumerate(E):
			DE[i, j] = min(DV[i, k], DV[i, l])
	return DE

def distance_similarity(D : npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
	ROW, COL = D.shape
	B : npt.NDArray[np.bool_] = np.empty((ROW, (COL - 1) * COL // 2), dtype=np.bool_)
	i = 0
	for j in range(COL):
		for k in range(j + 1, COL):
			B[:, i] = D[:, j] == D[:, k]
			i += 1
	return B

def reduced_distance_similarity(B : npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
	# Remove duplicates
	P = np.unique(B, axis=1)
	# Remove subsets
	COL = P.shape[1]
	keep = np.ones(COL).astype(np.bool_)
	for j in range(COL):
		P_j = P[:, [j]]
		P_other = np.delete(P, j, axis=1)
		keep_other = np.delete(keep, j)
		P_other = P_other[:, keep_other]
		is_subset = np.all((P_j | P_other) == P_other, axis=0)
		if np.any(is_subset):
			keep[j] = False
	P = P[:, keep]
	return P

def find_config_bruteforce(P : npt.NDArray[np.bool_]) -> Tuple[Any, ...]:
	nV : int = P.shape[0]
	config = (P, nV)
	return config

def find_config_boolean_satisfiability(P : npt.NDArray[np.bool_]) -> Tuple[Any, ...]:
	nV : int = P.shape[0]
	X = np.array([z3.Bool('x{}'.format(v + 1)) for v in range(nV)]) # pyright: ignore
	s = z3.Solver()
	if P.size > 0:
		s.add(z3.Or(*X)) # pyright: ignore
	for P_j in P.T:
		s.add(z3.Implies(z3.Or(*X[P_j]), z3.Or(*X[~P_j]))) # pyright: ignore
	config = (P, nV, X, s)
	return config

def find_config_linear_integer_arithmetic(P : npt.NDArray[np.bool_]) -> Tuple[Any, ...]:
	nV : int = P.shape[0]
	X : npt.NDArray[np.object_] = np.array([z3.Int('x{}'.format(v + 1)) for v in range(nV)]) # pyright: ignore
	config = (P, nV, X)
	return config

def find_config_pseudo_boolean(P : npt.NDArray[np.bool_]) -> Tuple[Any, ...]:
	nV : int = P.shape[0]
	nV = P.shape[0]
	X = np.array([z3.Bool('x{}'.format(v + 1)) for v in range(nV)]) # pyright: ignore
	X1 = np.array([(x, 1) for x in X])
	s = z3.Solver()
	config = (P, nV, X, X1, s)
	return config

def find_exact_bruteforce(config : Tuple[Any, ...], k : int) -> bool:
	P, nV = cast(Tuple[npt.NDArray[np.bool_], int], config)
	_P = P[:, P.sum(axis=0) >= k]
	combinations = itertools.combinations(range(nV), k)
	for indices in combinations:
		W = np.zeros((nV, 1), dtype=np.bool_)
		W[indices, 0] = True
		is_subset = np.all((W | _P) == _P, axis=0)
		if not np.any(is_subset):
			return True
	return False

def find_exact_boolean_satisfiability(config : Tuple[Any, ...], k : int) -> bool:
	_, _, X, s = cast(Tuple[Any, Any, npt.NDArray[np.object_], z3.Solver], config)
	s.push()
	s.add(z3.AtLeast(*X, k)) # pyright: ignore
	s.add(z3.AtMost(*X, k)) # pyright: ignore
	found = cast(bool, s.check() == z3.sat) # pyright: ignore
	s.pop()
	return found

def find_exact_linear_integer_arithmetic(config : Tuple[Any, ...], k : int) -> bool:
	P, _, X = cast(Tuple[npt.NDArray[np.bool_], Any, npt.NDArray[np.object_]], config)
	s = z3.Solver()
	for x in X:
		s.add(x >= 0) # pyright: ignore
		s.add(x <= 1) # pyright: ignore
	s.add(z3.Sum(*X) == k) # pyright: ignore
	_P = P[:, P.sum(axis=0) >= k]
	for _P_j in _P.T:
		s.add(z3.Sum(*X[_P_j]) <= k - 1) # pyright: ignore
	found = cast(bool, s.check() == z3.sat) # pyright: ignore
	return found

def find_exact_pseudo_boolean(config : Tuple[Any, ...], k : int) -> bool:
	P, _, _, X1, s = cast(Tuple[npt.NDArray[np.bool_], Any, Any, npt.NDArray[np.object_], z3.Solver], config)
	s.push()
	s.add(z3.PbEq(X1, k)) # pyright: ignore
	_P = P[:, P.sum(axis=0) >= k]
	if k == 0 and np.any(~np.any(_P, axis=0)):
		s.add(False) # pyright: ignore
	else:
		for _P_j in _P.T:
			s.add(z3.PbLe(X1[_P_j], k - 1)) # pyright: ignore
	found = cast(bool, s.check() == z3.sat) # pyright: ignore
	s.pop()
	return found

def find_bruteforce(P : npt.NDArray[np.bool_]) -> int:
	config = find_config_bruteforce(P)
	_, nV = cast(Tuple[Any, int], config)
	for k in range(nV - 1, -1, -1):
		found = find_exact_bruteforce(config, k)
		if not found:
			return k + 1
	return 0

def find_boolean_satisfiability(P : npt.NDArray[np.bool_]) -> int:
	config = find_config_boolean_satisfiability(P)
	_, nV, _, _ = cast(Tuple[Any, int, Any, Any], config)
	for k in range(nV - 1, -1, -1):
		found = find_exact_boolean_satisfiability(config, k) # pyright: ignore
		if not found:
			return k + 1
	return 0

def find_linear_integer_arithmetic(P : npt.NDArray[np.bool_]) -> int:
	config = find_config_linear_integer_arithmetic(P)
	_, nV, _ = cast(Tuple[Any, int, Any], config)
	for k in range(nV - 1, -1, -1):
		found = find_exact_linear_integer_arithmetic(config, k)
		if not found:
			return k + 1
	return 0

def find_pseudo_boolean(P : npt.NDArray[np.bool_]) -> int:
	config = find_config_pseudo_boolean(P)
	_, nV, _, _, _ = cast(Tuple[Any, int, Any, Any, Any], config)
	for k in range(nV - 1, -1, -1):
		found = find_exact_pseudo_boolean(config, k)
		if not found:
			return k + 1
	return 0
