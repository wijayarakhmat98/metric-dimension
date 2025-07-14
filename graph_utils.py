#!/usr/bin/env python3.12

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import floyd_warshall, connected_components

def graph6_decode(s):
	return nx.to_numpy_array(nx.from_graph6_bytes(s.encode()), dtype=bool).T

def graph6_encode(m):
	return nx.to_graph6_bytes(nx.from_numpy_array(m.T)).decode().strip()

def is_undirected(m):
	return np.array_equal(m, m.T)

def is_connected(m, directed=False):
	n_components, _ = connected_components(m, directed=directed)
	return n_components == 1

def distance_matrix(m, directed=False, connected=True, weighted=False):
	d = floyd_warshall(m.T, directed)
	if weighted:
		return d
	if not connected:
		d[np.isinf(d)] = -1
	return d.astype(int)

def edge_distance_matrix(m, d, directed=False, weighted=False):
	if not directed:
		m = m.copy()
		m[np.tril_indices_from(m)] = False
	l = np.argwhere(m)
	e = np.zeros((m.sum(), d.shape[0]), dtype=np.float64)
	for i in range(d.shape[0]):
		for r, (j, k) in enumerate(l):
			e[r, i] = min(d[j, i], d[k, i])
	if weighted:
		return e, l
	return e.astype(int), l
