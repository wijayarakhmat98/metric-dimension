import json
import multiprocessing
from pathlib import Path
import re
from typing import Any, cast, Dict, Optional, Set, Tuple
from utils import read_file

preserve_order = False
header = None

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		datum = cast(Dict[str, Any], json.loads(result))
		return datum
	except:
		return None

config_cache : Dict[int, Dict[str, Any]] = {}

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	_, exclude = cast(Tuple[Path, Set[str]], option)
	graph = cast(str, datum['graph'])
	if graph in exclude:
		return None
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum)

option_spec = [
	(['-c', '--log'], '', Path)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	log_filename, = cast(Tuple[Path], option)
	if not log_filename:
		return False
	return True

def option_augment(option : Tuple[Any, ...]) -> Tuple[Any, ...]:
	log_filename, = cast(Tuple[Path], option)
	with open(log_filename, 'r') as log_file:
		log_read = read_file(log_file)
		with multiprocessing.Pool() as pool:
			exclude = {graph for graph in set(pool.imap_unordered(extract_graph, log_read)) if graph}
	option = (*option, exclude)
	return option

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		usage:
			... <-c=<filename>>

		options:
			-c=<filename>, --log=<filename>
				Skip graphs that are already within the log file.
	'''
	).strip()

def extract_graph(result : str) -> Optional[str]:
	datum = decode(result)
	if not datum:
		return None
	graph = cast(str, datum['graph'])
	return graph
