#!/usr/bin/env python3.12

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

def graph6_decode(s):
	return nx.to_numpy_array(nx.from_graph6_bytes(s.encode()), dtype=np.bool_)

def graph6_encode(m):
	return nx.to_graph6_bytes(nx.from_numpy_array(m)).decode().strip()

def distance_matrix(m):
	return floyd_warshall(m, False).astype(np.int_)

def edge_distance_matrix(m, d):
	m = m.copy()
	m[np.tril_indices_from(m)] = np.False_
	l = np.argwhere(m)
	e = np.empty((m.sum(), d.shape[0]), dtype=np.int_)
	for i in range(d.shape[0]):
		for r, (j, k) in enumerate(l):
			e[r, i] = min(d[j, i], d[k, i])
	return e, l
