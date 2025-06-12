#!/usr/bin/env python3.13

import numpy as np

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
	return broadcast[mask].reshape(-1, n)
