from abc import ABC, abstractmethod
import itertools
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy as sp # type: ignore
from typing import cast
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

class find(ABC):
	P : npt.NDArray[np.bool_]
	nV : int

	def __init__(self, P : npt.NDArray[np.bool_]) -> None:
		self.P = P
		self.nV = P.shape[0]

	@abstractmethod
	def exact(self, k : int) -> bool:
		pass

	def minimum(self) -> int:
		for k in range(self.nV - 1, -1, -1):
			found = self.exact(k)
			if not found:
				return k + 1
		return 0

class find_bruteforce(find):
	def __init__(self, P : npt.NDArray[np.bool_]) -> None:
		super().__init__(P)

	def exact(self, k : int) -> bool:
		_P = self.P[:, self.P.sum(axis=0) >= k]
		combinations = itertools.combinations(range(self.nV), k)
		for indices in combinations:
			W = np.zeros((self.nV, 1), dtype=np.bool_)
			W[indices, 0] = True
			is_subset = np.all((W | _P) == _P, axis=0)
			if not np.any(is_subset):
				return True
		return False

class find_boolean_satisfiability(find):
	X : npt.NDArray[np.object_]
	s : z3.Solver

	def __init__(self, P : npt.NDArray[np.bool_]) -> None:
		super().__init__(P)
		self.X = np.array([z3.Bool('x{}'.format(v + 1)) for v in range(self.nV)]) # pyright: ignore
		self.s = z3.Solver()
		if self.P.size > 0:
			self.s.add(z3.Or(*self.X)) # pyright: ignore
		for P_j in self.P.T:
			self.s.add(z3.Implies(z3.Or(*self.X[P_j]), z3.Or(*self.X[~P_j]))) # pyright: ignore

	def exact(self, k : int) -> bool:
		self.s.push()
		self.s.add(z3.AtLeast(*self.X, k)) # pyright: ignore
		self.s.add(z3.AtMost(*self.X, k)) # pyright: ignore
		found = cast(bool, self.s.check() == z3.sat) # pyright: ignore
		self.s.pop()
		return found

class find_linear_integer_arithmetic(find):
	X : npt.NDArray[np.object_]

	def __init__(self, P : npt.NDArray[np.bool_]) -> None:
		super().__init__(P)
		self.X = np.array([z3.Int('x{}'.format(v + 1)) for v in range(self.nV)]) # pyright: ignore

	def exact(self, k : int) -> bool:
		s = z3.Solver()
		for x in self.X:
			s.add(x >= 0) # pyright: ignore
			s.add(x <= 1) # pyright: ignore
		s.add(z3.Sum(*self.X) == k) # pyright: ignore
		_P = self.P[:, self.P.sum(axis=0) >= k]
		for _P_j in _P.T:
			s.add(z3.Sum(*self.X[_P_j]) <= k - 1) # pyright: ignore
		found = cast(bool, s.check() == z3.sat) # pyright: ignore
		return found

class find_pseudo_boolean(find):
	X : npt.NDArray[np.object_]
	X1 : npt.NDArray[np.object_]
	s : z3.Solver

	def __init__(self, P : npt.NDArray[np.bool_]) -> None:
		super().__init__(P)
		self.X = np.array([z3.Bool('x{}'.format(v + 1)) for v in range(self.nV)]) # pyright: ignore
		self.X1 = np.array([(x, 1) for x in self.X])
		self.s = z3.Solver()

	def exact(self, k : int) -> bool:
		self.s.push()
		self.s.add(z3.PbEq(self.X1, k)) # pyright: ignore
		_P = self.P[:, self.P.sum(axis=0) >= k]
		if k == 0 and np.any(~np.any(_P, axis=0)):
			self.s.add(False) # pyright: ignore
		else:
			for _P_j in _P.T:
				self.s.add(z3.PbLe(self.X1[_P_j], k - 1)) # pyright: ignore
		found = cast(bool, self.s.check() == z3.sat) # pyright: ignore
		self.s.pop()
		return found

ALGORITHMS = (
	'bruteforce',
	'boolean_satisfiability',
	'linear_integer_arithmetic',
	'pseudo_boolean'
)

def create_find(P : npt.NDArray[np.bool_], method : str) -> find:
	match method:
		case 'bruteforce':
			return find_bruteforce(P)
		case 'boolean_satisfiability':
			return find_boolean_satisfiability(P)
		case 'linear_integer_arithmetic':
			return find_linear_integer_arithmetic(P)
		case 'pseudo_boolean':
			return find_pseudo_boolean(P)
		case _:
			raise AssertionError
