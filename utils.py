import base64
import copy
import hashlib
import json
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph # pyright: ignore
import re
from typing import Any, cast, Dict, List

def file_to_list(filename : str) -> List[str]:
	with open(filename, 'r') as file:
		lines = [line.strip() for line in file]
	return lines

def hash(s : str) -> str:
	sha256_hash = hashlib.sha256(s.encode()).digest()
	b64_encoded = base64.urlsafe_b64encode(sha256_hash).decode()
	return b64_encoded[:10]

def graph6_draw(s : str, filename : str) -> None:
	b = s.encode()
	g = nx.from_graph6_bytes(b) # pyright: ignore
	mapping : Dict[int, str] = {i: 'x{}'.format(i + 1) for i in g.nodes} # pyright: ignore
	g = cast(nx.Graph, nx.relabel_nodes(g, mapping)) # type: ignore
	a = to_agraph(g) # pyright: ignore
	a.graph_attr.update(splines='true') # pyright: ignore
	a.layout('fdp') # pyright: ignore
	a.draw(filename) # pyright: ignore

def info_stringify(info : Dict[str, Any]) -> str:
	info = copy.deepcopy(info)
	info['vertex']['enumerate']['resolving'] = [str(s) for s in info['vertex']['enumerate']['resolving']]
	info['edge']['enumerate']['resolving'] = [str(s) for s in info['edge']['enumerate']['resolving']]
	s = json.dumps(info, indent=2)
	s = re.sub(r'"({\'set.*)"', lambda m: re.sub(r"'", r'"', m.group(1)), s)
	s = re.sub('True', 'true', s)
	s = re.sub('False', 'false', s)
	return s
