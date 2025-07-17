#!/usr/bin/env python3.13

import base64
import hashlib
import importlib
import multiprocessing
import metric_dimension
import re
import subprocess
import sys
from types import ModuleType
from typing import cast, Dict, List, Protocol, TextIO, Tuple

def main(args : List[str]) -> None:
	mypy(__file__)
	mypy('metric_dimension.py')
	metric_dimension.__njit_compile__()
	if len(args) < 1:
		main_usage()
	match args[0]:
		case 'debug'  : main_debug(args[1:])
		case 'process': main_process(args[1:])
		case _        : main_usage()

def main_usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				debug <graph6 string> [--module=<debug module>]

				process [--resume=<result file>] [--module=<process module>]
					Read graphs from stdin.
		'''
		).strip()
	)
	sys.exit()

class ProtocolDebug(Protocol):
	def debug(self, graph : str) -> None: ...

def main_debug(args : List[str]) -> None:
	if len(args) < 1:
		main_usage()
	graph = args[0]
	option = parse_args(args[1:], [
		('--module', 'module_path', 'default_debug.py')
	])
	debug = cast(ProtocolDebug, load_module(option['module_path']))
	debug.debug(graph)

class ProtocolProcess(Protocol):
	def process(self, graph : str) -> str: ...
	def resume(self, s : str) -> str: ...

def main_process(args : List[str]) -> None:
	option = parse_args(args, [
		('--resume', 'filename', ''),
		('--module', 'module_path', 'default_process.py')
	])
	process = cast(ProtocolProcess, load_module(option['module_path']))
	graphs = read_file(sys.stdin)
	if option['filename']:
		with open(option['filename'], 'r') as file:
			result_exclude = read_file(file)
		with multiprocessing.Pool() as pool:
			graphs_exclude = set(pool.imap_unordered(process.resume, result_exclude))
		graphs = list(set(graphs) - graphs_exclude)
	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(process.process, graphs):
			print(result)

def hash(s : str) -> str:
	sha256_hash = hashlib.sha256(s.encode()).digest()
	b64_encoded = base64.urlsafe_b64encode(sha256_hash).decode()
	return b64_encoded[:10]

def read_file(file : TextIO) -> List[str]:
	return [line for raw in file if (line := raw.strip())]

def load_module(module_path : str) -> ModuleType:
	module_name = hash(module_path)
	mypy(module_path)
	spec = importlib.util.spec_from_file_location(module_name, module_path) # type: ignore
	module = cast(ModuleType, importlib.util.module_from_spec(spec)) # type: ignore
	sys.modules[module_name] = module
	spec.loader.exec_module(module) # pyright: ignore
	return module

def parse_args(args : List[str], config : List[Tuple[str, str, str]]) -> Dict[str, str]:
	map : Dict[str, str] = {}
	option : Dict[str, str] = {}
	for flag, key, value in config:
		map[flag] = key
		option[key] = value
	for arg in args:
		key, value = [s.strip() for s in (arg.split('=', 1) + [''])[:2]]
		if key in map and value:
			option[map[key]] = value
	return option

def mypy(filename: str) -> None:
	result = subprocess.run(['mypy', filename], capture_output=True, text=True)
	if result.returncode != 0:
		print(result.stdout)
		print(result.stderr, file=sys.stderr)
		sys.exit()

if __name__ == '__main__':
	main(sys.argv[1:])
