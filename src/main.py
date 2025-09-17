#!/usr/bin/env python3.13

import base64
import hashlib
from functools import partial
import importlib
import multiprocessing
import metric_dimension
import re
import sys
from types import ModuleType
from typing import Any, cast, Dict, Iterator, List, Protocol, Self, Set, TextIO
from utils import parse_args

def main(args : List[str]) -> None:
	metric_dimension.__njit_compile__()
	if len(args) < 1:
		main_usage()
	match args[0]:
		case 'd' | 'debug'  : main_debug(args[1:])
		case 'p' | 'process': main_process(args[1:])
		case 'f' | 'format' : main_format(args[1:])
		case _        : main_usage()

def main_usage() -> None:
	print(
		re.sub(r'\n\t\t\t', r'\n',
		'''
			usage:
				d|debug <graph6 string> [-m=<debug module>]

				p|process [--resume=<result file>] [-m=<process module>]
					Read graphs from stdin.

				f|format [-s] [-m=<format module>]
					Read results from stdin.

			options:
				-m=<module name>, --module=<module name>

				-h, --help
					A module may have a help page.
		'''
		).strip(),
		file=sys.stderr
	)
	sys.exit()

class ProtocolDebug(Protocol):
	def debug(self, graph : str, option_raw : List[str]) -> None: ...

def main_debug(args : List[str]) -> None:
	if len(args) < 1:
		main_usage()
	graph = args[0]
	option = parse_args(args[1:], [
		(['-m', '--module'], 'module_name', 'b')
	])
	debug = cast(ProtocolDebug, load_module('debug.{}'.format(option['module_name'])))
	debug.debug(graph, args)

class ProtocolProcess(Protocol):
	def process(self, graph : str, option_raw : List[str]) -> str: ...
	def resume(self, result : str, option_raw : List[str]) -> str: ...

def main_process(args : List[str]) -> None:
	option = parse_args(args, [
		(['--resume'], 'filename', ''),
		(['-m', '--module'], 'module_name', 'identity')
	])
	process = cast(ProtocolProcess, load_module('process.{}'.format(option['module_name'])))
	graphs = read_file(sys.stdin)
	if option['filename']:
		with open(option['filename'], 'r') as file:
			results = read_file(file)
			with multiprocessing.Pool() as pool:
				graphs_exclude = set(pool.imap_unordered(partial(process.resume, option_raw=args), results))
		graphs.set_exclude(graphs_exclude)
	with multiprocessing.Pool() as pool:
		for result in pool.imap_unordered(partial(process.process, option_raw=args), graphs):
			print(result)

class ProtocolFormat(Protocol):
	def decode(self, s : str) -> Dict[str, Any]: ...
	def format(self, results : Iterator[str], option_raw : List[str]) -> None: ...

def main_format(args : List[str]) -> None:
	option = parse_args(args, [
		(['-m', '--module'], 'module_name', 'identity')
	])
	format = cast(ProtocolFormat, load_module('format.{}'.format(option['module_name'])))
	results = read_file(sys.stdin)
	format.format(results, args)

def hash(s : str) -> str:
	sha256_hash = hashlib.sha256(s.encode()).digest()
	b64_encoded = base64.urlsafe_b64encode(sha256_hash).decode()
	return b64_encoded[:10]

class read_file(Iterator[str]):
	def __init__(self, file : TextIO, exclude : Set[str] = set()):
		self.file = file
		self.exclude = exclude

	def __iter__(self) -> Self:
		return self

	def __next__(self) -> str:
		for raw in self.file:
			line = raw.strip()
			if not line:
				continue
			if line in self.exclude:
				continue
			return line
		raise StopIteration

	def set_exclude(self, exclude : Set[str]) -> Self:
		self.exclude = exclude
		return self

def load_module(module_name : str) -> ModuleType:
	module = importlib.import_module(module_name)
	return module

if __name__ == '__main__':
	main(sys.argv[1:])
