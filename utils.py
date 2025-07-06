#!/usr/bin/env python3.13

import copy
import hashlib
import json
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import re

def file_to_list(filename):
	with open(filename, 'r') as file:
		ss = file.readlines()
		ss = [s.strip() for s in ss]
		return ss

def hash(s):
	return hashlib.sha256(s.encode()).hexdigest()

def graph6_draw(s, filename):
	g = nx.from_graph6_bytes(s.encode())
	mapping = {i: 'x{}'.format(i + 1) for i in g.nodes}
	g = nx.relabel_nodes(g, mapping)
	a = to_agraph(g)
	a.graph_attr.update(
			splines='true'
	)
	a.layout('fdp')
	a.draw(filename)

def info_stringify(info):
	infos = copy.deepcopy(info)
	infos['vertex']['enumerate']['resolving'] = [str(s) for s in infos['vertex']['enumerate']['resolving']]
	infos['edge']['enumerate']['resolving'] = [str(s) for s in infos['edge']['enumerate']['resolving']]
	infos = json.dumps(infos, indent=2)
	infos = re.sub(r'"({\'set.*)"', lambda m: re.sub(r"'", r'"', m.group(1)), infos)
	return infos
