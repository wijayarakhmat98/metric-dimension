#!/usr/bin/env python3.13

import networkx as nx

def str_to_graph(s):
	return nx.from_graph6_bytes(s.encode())

def graph_to_matrix(g):
	return nx.to_numpy_array(g, dtype=bool)

def str_to_matrix(s):
	return graph_to_matrix(str_to_graph(s))

def map_str_to_matrix(ss):
	return {s: str_to_matrix(s) for s in ss}

def map_file_to_matrix(filename):
	with open(filename, 'r') as file:
		ss = file.readlines()
		ss = [s.strip() for s in ss]
	return map_str_to_matrix(ss)
