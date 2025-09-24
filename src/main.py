#!/usr/bin/env python3.13

from functools import partial
import importlib
import multiprocessing
import metric_dimension
import re
import sys
from types import ModuleType
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Protocol, Self, TextIO, Tuple
from utils import parse_args, parse_switch

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
					Prints the module help page.
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
	module_name, = cast(
		Tuple[str],
		parse_args(args[1:], [
			(['-m', '--module'], 'b', None)
		])
	)
	debug = cast(ProtocolDebug, load_module('debug.{}'.format(module_name)))
	debug.debug(graph, args)

class ProtocolProcess(Protocol):
	preserve_order : bool
	header : Optional[Callable[[Dict[str, Any]], str]]
	decode : Callable[[str], Dict[str, Any]]
	make_transform : Callable[[Tuple[Any, ...]], Callable[[Dict[str, Any]], Dict[str, Any]]]
	encode : Callable[[Dict[str, Any]], str]
	option_spec : Optional[List[Tuple[List[str], Any, Optional[Callable[[str], Any]]]]]
	option_valid : Optional[Callable[[Tuple[Any, ...]], bool]]
	help : Callable[[], str]

process : Optional[ProtocolProcess] = None
process_transform : Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

def process_compose(result : str, module_name : str, args : List[str]) -> Tuple[Dict[str, Any], str]:
	global process, process_transform
	if process is None:
		process = cast(ProtocolProcess, load_module('process.{}'.format(module_name)))
	if process_transform is None:
		if process.option_spec is None:
			process_option = cast(Tuple[Any, ...], ())
		else:
			process_option = parse_args(args, process.option_spec)
		process_transform = process.make_transform(process_option)
	datum = process.decode(result)
	datum = process_transform(datum)
	result = process.encode(datum)
	return datum, result

def main_process(args : List[str]) -> None:
	help, module_name = cast(
		Tuple[bool, str],
		parse_args(args, [
			(['-h', '--help'], False, parse_switch),
			(['-m', '--module'], 'identity', None)
		])
	)
	bound_process_compose = partial(process_compose, module_name=module_name, args=args)
	process = cast(ProtocolProcess, load_module('process.{}'.format(module_name)))
	if help:
		print(process.help(), file=sys.stderr)
		return
	if process.option_spec is not None:
		process_option = parse_args(args, process.option_spec)
		if process.option_valid is not None and not process.option_valid(process_option):
			print(process.help(), file=sys.stderr)
			return
	results = read_file(sys.stdin)
	with multiprocessing.Pool() as pool:
		if process.header is None:
			if process.preserve_order:
				for _, result in pool.imap(bound_process_compose, results):
					print(result)
			else:
				for _, result in pool.imap_unordered(bound_process_compose, results):
					print(result)
		else:
			print_header = True
			if process.preserve_order:
				for datum, result in pool.imap(bound_process_compose, results):
					if print_header:
						print(process.header(datum))
						print_header = False
					print(result)
			else:
				for datum, result in pool.imap_unordered(bound_process_compose, results):
					if print_header:
						print(process.header(datum))
						print_header = False
					print(result)

class ProtocolFormat(Protocol):
	def decode(self, s : str) -> Dict[str, Any]: ...
	def format(self, results : Iterator[str], option_raw : List[str]) -> None: ...

def main_format(args : List[str]) -> None:
	module_name, = cast(
		Tuple[str],
		parse_args(args, [
			(['-m', '--module'], 'sort', None)
		])
	)
	format = cast(ProtocolFormat, load_module('format.{}'.format(module_name)))
	results = read_file(sys.stdin)
	format.format(results, args)

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
