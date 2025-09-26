import json
import math
import matplotlib
import matplotlib.pyplot as plt
import metric_dimension
import networkx as nx
import numpy as np
import numpy.typing as npt
from pathlib import Path
import re
from typing import Any, Callable, cast, Dict, Tuple
from utils import hash

preserve_order = False
header = None

def decode(result : str) -> Dict[str, Any]:
	datum = cast(Dict[str, Any], json.loads(result))
	return datum

config_cache : Dict[int, Dict[str, Any]] = {}

def make_transform(option : Tuple[Any, ...]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
	path, = cast(Tuple[Path], option)
	path.mkdir(parents=True, exist_ok=True)
	matplotlib.use('Agg')
	def transform(datum : Dict[str, Any]) -> Dict[str, Any]:
		if 'hash' in datum:
			h = cast(str, datum['hash'])
		else:
			h = hash(datum['graph'])
			datum['hash'] = h
		path_img = path / '{}.svg'.format(datum['hash'])
		n, m = metric_dimension.graph6_decode(datum['graph'])
		if n not in config_cache:
			config_cache[n] = config(n)
		draw(m, path_img, config_cache[n])
		return datum
	return transform

def encode(datum : Dict[str, Any]) -> str:
	return json.dumps(datum)

option_spec = [
	(['-f', '--folder'], '', Path)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	path, = cast(Tuple[Path], option)
	if not path:
		return False
	return True

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		Create illustration for each graph.

		usage:
			... <-f=<pattern>>

		options:
			-f=<path>, --folder=<path>
				Write output to this location.
	'''
	).strip()

def config(n : int) -> Dict[str, Any]:
	font_size = 12
	node_r = (len(str(n)) + 1) * font_size / 2
	r = max(2 * n * node_r / math.pi, 3 * node_r)
	x_d = 2 * r + 4 * node_r
	y_d = 2 * r + 4 * node_r
	vs = ['x{}'.format(i + 1) for i in range(n)]
	fig_size = (x_d / 72, y_d / 72)
	ax_xbound = (x_d / -2, x_d / 2)
	ax_ybound = (y_d / -2, y_d / 2)
	node_size = (2 * node_r) ** 2
	pos = {
		v: (
			r * math.cos(math.pi / 2 - (i + 1) * 2 * math.pi / n),
			r * math.sin(math.pi / 2 - (i + 1) * 2 * math.pi / n)
		)
		for i, v in enumerate(vs)
	}
	label = {i: v for i, v in enumerate(vs)}
	config = {
		'font_size': font_size,
		'fig_size': fig_size,
		'ax_xbound': ax_xbound,
		'ax_ybound': ax_ybound,
		'node_size': node_size,
		'pos': pos,
		'label': label
	}
	return config

def draw(m : npt.NDArray[np.bool_], path_img : Path, config : Dict[str, Any]) -> None:
	_, ax = plt.subplots(figsize=config['fig_size']) # pyright: ignore
	ax.set_xlim(*config['ax_xbound'])
	ax.set_ylim(*config['ax_ybound'])
	g = nx.from_numpy_array(m)
	g = nx.relabel_nodes(g, config['label'])
	nx.draw(g, pos=config['pos'], with_labels=True, font_size=config['font_size'], node_size=config['node_size'], node_color='white', edgecolors='black') # pyright: ignore
	plt.savefig(path_img, bbox_inches='tight', pad_inches=0) # pyright: ignore
	plt.close()
