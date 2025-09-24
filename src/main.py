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
from typing import Any, cast, Dict, Iterator, List, Optional, Protocol, Self, TextIO
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

				p|process [-m=<process module>]
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
	preserve_order : bool
	def decode(self, result : str) -> Dict[str, Any]: ...
	def transform(self, datum : Dict[str, Any]) -> Dict[str, Any]: ...
	def encode(self, datum : Dict[str, Any]) -> str: ...

process : Optional[ProtocolProcess] = None

def process_compose(result : str, module_name : str) -> str:
	global process
	if process is None:
		process = cast(ProtocolProcess, load_module('process.{}'.format(module_name)))
	datum = process.decode(result)
	datum = process.transform(datum)
	result = process.encode(datum)
	return result

def main_process(args : List[str]) -> None:
	option = parse_args(args, [
		(['-m', '--module'], 'module_name', 'identity')
	])
	module_name = option['module_name']
	bound_process_compose = partial(process_compose, module_name=module_name)
	process = cast(ProtocolProcess, load_module('process.{}'.format(module_name)))
	results = read_file(sys.stdin)
	with multiprocessing.Pool() as pool:
		if process.preserve_order:
			for result in pool.imap(bound_process_compose, results):
				print(result)
		else:
			for result in pool.imap_unordered(bound_process_compose, results):
				print(result)

class ProtocolFormat(Protocol):
	def decode(self, s : str) -> Dict[str, Any]: ...
	def format(self, results : Iterator[str], option_raw : List[str]) -> None: ...

def main_format(args : List[str]) -> None:
	option = parse_args(args, [
		(['-m', '--module'], 'module_name', 'sort')
	])
	format = cast(ProtocolFormat, load_module('format.{}'.format(option['module_name'])))
	results = read_file(sys.stdin)
	format.format(results, args)

def hash(s : str) -> str:
	sha256_hash = hashlib.sha256(s.encode()).digest()
	b64_encoded = base64.urlsafe_b64encode(sha256_hash).decode()
	return b64_encoded[:10]

class read_file(Iterator[str]):
	def __init__(self, file : TextIO):
		self.file = file

	def __iter__(self) -> Self:
		return self

	def __next__(self) -> str:
		for raw in self.file:
			line = raw.strip()
			if not line:
				continue
			return line
		raise StopIteration

def load_module(module_name : str) -> ModuleType:
	module = importlib.import_module(module_name)
	return module

if __name__ == '__main__':
	main(sys.argv[1:])
