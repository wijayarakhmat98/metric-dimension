#!/usr/bin/env python3.12

import numpy as np
import z3

def create_node_boolean(m):
	n = m.shape[0]
	return np.array([z3.Bool('x{}'.format(i+1)) for i in range(n)])

def distance_similarity_broadcast(d):
	return d[None, :, :] == d[:, None, :]

def distance_similarity_prune(b):
	p = np.unique(b.reshape(-1, b.shape[2]), axis=0)
	p = p[~np.all(p, axis=1)]
	return p

def distance_similarity_permute(p):
	t = [p]
	for c in p:
		idx = np.where(c)[0]
		n = len(idx)
		mask = np.arange(2**n)[:, None] & (1 << np.arange(n)) == 0
		a = np.tile(c, (2**n, 1))
		a[np.arange(2**n)[:, None], idx] &= mask
		t.append(a)
	t = np.unique(np.vstack(t), axis=0)
	return t

def distance_similarity_group(t):
	u = [[] for _ in range(t.shape[1] + 1)]
	for c in t:
		u[np.sum(c)].append(c)
	for i in range(len(u)):
		if len(u[i]) > 0:
			u[i] = np.vstack(u[i])
	return u

def apply_boolean_similarity(v, g):
	return [z3.And(*v[c]) for c in g]

def find_exact(v, u, n):
	s = z3.Solver()
	s.add(z3.Not(z3.Or(apply_boolean_similarity(v, u[n]))))
	s.add(z3.AtLeast(*v, n))
	s.add(z3.AtMost(*v, n))
	if s.check() == z3.unsat:
		return []
	model = s.model()
	w = [b for b in v if z3.is_true(model.evaluate(b))]
	return w

def find_least(v, u):
	for i in range(1, len(u)):
		w = find_exact(v, u, i)
		if len(w) != 0:
			return w
	return []

def find_enumerate(v, u, n):
	s = z3.Solver()
	s.add(z3.Not(z3.Or(apply_boolean_similarity(v, u[n]))))
	s.add(z3.AtLeast(*v, n))
	s.add(z3.AtMost(*v, n))
	ws = []
	while s.check() == z3.sat:
		model = s.model()
		w = [b for b in v if z3.is_true(model.evaluate(b))]
		s.add(z3.Not(z3.And(w)))
		ws.append(w)
	return ws

def resolving_representation(v, w, d):
	return d[:, np.array([b in w for b in v])]

def valid(r):
	unique, idx = np.unique(r, axis=0, return_index=True)
	unique = unique[np.argsort(idx)]
	return np.array_equal(r, unique)
