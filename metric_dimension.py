#!/usr/bin/env python3.13

import numpy as np
import z3

def distance_similarity_broadcast(d):
	return d[None, :, :] == d[:, None, :]

def distance_similarity_mask(d):
	n = d.shape[0]
	return np.broadcast_to(
		(np.arange(n)[None, :] > np.arange(n)[:, None])[:, :, None],
		(n, n, n)
	)

def distance_similarity_prune(d):
	n = d.shape[0]
	broadcast = distance_similarity_broadcast(d)
	mask = distance_similarity_mask(d)
	prune = broadcast[mask].reshape(-1, n)
	prune = np.unique(prune, axis=0)
	prune = prune[np.any(prune, axis=1)]
	return prune

def create_node_boolean(m):
	n = m.shape[0]
	return np.array([z3.Bool('x{}'.format(i+1)) for i in range(n)])

def apply_boolean_similarity(v, p):
	return [z3.And(*v[mask]) for mask in p]

def find_least(v, constraint):
	count = z3.Sum(*[z3.If(b, 0, 1) for b in v])
	o = z3.Optimize()
	o.add(constraint)
	o.minimize(count)
	if o.check() != z3.sat:
		return z3.unsat, -1, []
	model = o.model()
	w = [b for b in v if z3.is_false(model.evaluate(b))]
	return z3.sat, len(w), w

def get_resolving_representation(v, d, w):
	mask = np.array([b in w for b in v])
	return d[:, mask]
