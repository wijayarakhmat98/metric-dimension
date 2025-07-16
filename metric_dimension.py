#!/usr/bin/env python3.12

import numpy as np
import z3

def create_node_boolean(m):
	n = m.shape[0]
	return np.array([z3.Int('x{}'.format(i+1)) for i in range(n)])

def distance_similarity_broadcast(d):
	return d[None, :, :] == d[:, None, :]

def distance_similarity_prune(b):
	p = np.unique(b.reshape(-1, b.shape[2]), axis=0)
	p = p[np.any(p, axis=1) & ~np.all(p, axis=1)]
	return p

def find_least(v, u):
	s = z3.Optimize()
	s.add(z3.And([z3.And(b >= 0, b <= 1) for b in v]))
	k = z3.Int('k')
	s.add(z3.Sum(*v) == k)
	s.add(k >= 1)
	s.add(z3.And([z3.Sum(*v[c]) < k for c in u]))
	s.minimize(k)
	s.check()
	return s.model()[k].as_long()

def find_enumerate(v, u, n):
	s = z3.Solver()
	s.add(z3.And([z3.And(b >= 0, b <= 1) for b in v]))
	s.add(z3.Sum(*v) == n)
	s.add(z3.And([z3.Sum(*v[c]) < n for c in u]))
	ws = []
	while s.check() == z3.sat:
		model = s.model()
		w = [model[b].as_long() for b in v]
		s.add(z3.Or([b != x for b, x in zip(v, w)]))
		ws.append(w)
	return np.array(ws, dtype=np.bool_)

def resolving_representation(v, w, d):
	return d[:, w]

def valid(r):
	unique, idx = np.unique(r, axis=0, return_index=True)
	unique = unique[np.argsort(idx)]
	return np.array_equal(r, unique)
