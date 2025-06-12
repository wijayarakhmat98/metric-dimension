#!/usr/bin/env python3.13

import networkx as nx

def str_to_graph(s):
	return nx.from_graph6_bytes(s.encode())

def graph_to_matrix(g):
	return nx.to_numpy_array(g, dtype=bool)

def str_to_matrix(s):
	return graph_to_matrix(str_to_graph(s))
