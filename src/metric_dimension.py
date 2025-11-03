import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy as sp # type: ignore
from typing import cast

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

def find_exact_bruteforce(P : npt.NDArray[np.bool_], k : int) -> bool:
	P = P[:, P.sum(axis=0) >= k]
	nV, COL = P.shape
	indices : npt.NDArray[np.int_] = np.arange(k)
	while True:
		W = np.zeros(nV).astype(np.bool_)
		for i in indices:
			W[i] = True
		for j in range(COL):
			if np.array_equal(W | P[:, j], P[:, j]):
				break
		else:
			return True
		for i in range(k - 1, -1, -1):
			if indices[i] != i + nV - k:
				break
		else:
			break
		indices[i] += 1
		for j in range(i + 1, k):
			indices[j] = indices[j - 1] + 1
	return False

def find_bruteforce(P : npt.NDArray[np.bool_]) -> int:
	nV = P.shape[0]
	for k in range(nV - 1, -1, -1):
		if not find_exact_bruteforce(P, k):
			return k + 1
	return -1
