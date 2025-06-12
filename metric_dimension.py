#!/usr/bin/env python3.13

def distance_similarity(d):
	return d[None, :, :] == d[:, None, :]
