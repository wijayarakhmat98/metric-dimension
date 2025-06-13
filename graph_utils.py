#!/usr/bin/env python3.13

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import floyd_warshall, connected_components

def graph6_decode(s):
	return nx.to_numpy_array(nx.from_graph6_bytes(s.encode()), dtype=bool)

def graph6_encode(m):
	return nx.to_graph6_bytes(nx.from_numpy_array(m.astype(int))).decode().strip()

def is_undirected(m):
	return np.array_equal(m, m.T)

def is_connected(m, directed=False):
	n_components, _ = connected_components(m, directed=directed)
	return n_components == 1

def distance_matrix(m, directed=False, connected=True, weighted=False):
	d = floyd_warshall(m, directed)
	if weighted:
		return d
	if not connected:
		d[np.isinf(d)] = -1
	return d.astype(int)
