#!/usr/bin/env python3.13

import base64
import hashlib
import importlib
import multiprocessing
import metric_dimension
import re
import sys
from types import ModuleType
from typing import Any, cast, Dict, List, Protocol, TextIO, Tuple

def main(args : List[str]) -> None:
	metric_dimension.__njit_compile__()
	if len(args) < 1:
		main_usage()
	match args[0]:
		case 'debug'  : main_debug(args[1:])
		case 'process': main_process(args[1:])
		case 'format' : main_format(args[1:])
		case _        : main_usage()

def main_usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				debug <graph6 string> [-m=<debug module>]

				process [--resume=<result file>] [-m=<process module>]
					Read graphs from stdin.

				format [-s=<true|false>] [-m=<format module>]
					Read results from stdin.

			options:
				-s=<true|false>, --sort=<true|false>
					Defaults to false.

				-m=<module name>, --module=<module name>

				-h, --help
					A module may have a help page.
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
		(['-m', '--module'], 'module_name', 'debug.b_be')
	])
	debug = cast(ProtocolDebug, load_module(option['module_name']))
	debug.debug(graph)

class ProtocolProcess(Protocol):
	def process(self, graph : str) -> str: ...
	def resume(self, s : str) -> str: ...

def main_process(args : List[str]) -> None:
	option = parse_args(args, [
		(['--resume'], 'filename', ''),
		(['-m', '--module'], 'module_name', 'process.b_wn_be_wen')
	])
	process = cast(ProtocolProcess, load_module(option['module_name']))
	graphs = read_file(sys.stdin)
	if option['filename']:
		with open(option['filename'], 'r') as file:
			results = read_file(file)
		with multiprocessing.Pool() as pool:
			graphs_exclude = set(pool.imap_unordered(process.resume, results))
		graphs = list(set(graphs) - graphs_exclude)
	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(process.process, graphs):
			print(result)

class ProtocolFormat(Protocol):
	def decode(self, s : str) -> Dict[str, Any]: ...
	def format(self, results : List[str], sort : bool, option : Dict[str, str]) -> None: ...

def main_format(args : List[str]) -> None:
	option = parse_args(args, [
		(['-s', '--sort'], 'sort', 'false'),
		(['-m', '--module'], 'module_name', 'format.indent')
	])
	format = cast(ProtocolFormat, load_module(option['module_name']))
	results = read_file(sys.stdin)
	format.format(results, option['sort'] == 'true', option)

def hash(s : str) -> str:
	sha256_hash = hashlib.sha256(s.encode()).digest()
	b64_encoded = base64.urlsafe_b64encode(sha256_hash).decode()
	return b64_encoded[:10]

def read_file(file : TextIO) -> List[str]:
	return [line for raw in file if (line := raw.strip())]

def load_module(module_name : str) -> ModuleType:
	module = importlib.import_module(module_name)
	return module

def parse_args(args : List[str], config : List[Tuple[List[str], str, str]]) -> Dict[str, str]:
	map : Dict[str, str] = {}
	option : Dict[str, str] = {}
	for flags, key, value in config:
		for flag in flags:
			map[flag] = key
		option[key] = value
	for arg in args:
		key, value = [s.strip() for s in (arg.split('=', 1) + [''])[:2]]
		if key in map and value:
			option[map[key]] = value
	return option

if __name__ == '__main__':
	main(sys.argv[1:])
