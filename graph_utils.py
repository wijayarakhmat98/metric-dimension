#!/usr/bin/env python3.13

import numpy as np
from scipy.sparse.csgraph import floyd_warshall, connected_components

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
