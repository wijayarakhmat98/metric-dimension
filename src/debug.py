#!/usr/bin/env python3.13

import metric_dimension
import re
import signal
import sys
from typing import cast, List, Protocol, Tuple
from utils import load_module, parse_args

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

class ProtocolDebug(Protocol):
	def debug(self, graph : str, option_raw : List[str]) -> None: ...

def main(args : List[str]) -> None:
	metric_dimension.__njit_compile__()
	if len(args) < 1:
		print(help(), file=sys.stderr)
		return
	graph = args[0]
	module_name, = cast(
		Tuple[str],
		parse_args(args[1:], [
			(['-m', '--module'], '', None)
		])
	)
	if not module_name:
		print(help(), file=sys.stderr)
		return
	debug = cast(ProtocolDebug, load_module('debug.{}'.format(module_name)))
	debug.debug(graph, args)

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		usage:
			<graph6 string> <-m=<debug module>>

		options:
			-m=<module name>, --module=<module name>

			-h, --help
				Prints the module help page.
	'''
	).strip()

if __name__ == '__main__':
	main(sys.argv[1:])
