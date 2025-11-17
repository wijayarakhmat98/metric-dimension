import json
import metric_dimension
import os
import re
from typing import Any, cast, Dict, Optional, Tuple, Union
from utils import parse_switch, timer, timeout, timeout_exception

preserve_order = False
header = None

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		datum = cast(Dict[str, Any], json.loads(result))
		return datum
	except:
		return None

UNSET = -1
TIMEOUT = -2
ERROR = -3

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	method, edge, limit = cast(Tuple[str, bool, float], option)
	if method not in ('bruteforce', 'boolean_satisfiability', 'linear_integer_arithmetic', 'pseudo_boolean'):
		return None
	s = datum['graph']
	M = metric_dimension.graph6_decode(s)
	DV = metric_dimension.distance_matrix(M)
	if edge:
		E = metric_dimension.edges(M)
		DE = metric_dimension.edge_distance_matrix(E, DV)
		D = DE
	else:
		D = DV
	B = metric_dimension.distance_similarity(D)
	P = metric_dimension.reduced_distance_similarity(B)
	find : metric_dimension.find
	match method:
		case 'bruteforce':
			find = metric_dimension.find_bruteforce(P)
		case 'boolean_satisfiability':
			find = metric_dimension.find_boolean_satisfiability(P)
		case 'linear_integer_arithmetic':
			find = metric_dimension.find_linear_integer_arithmetic(P)
		case 'pseudo_boolean':
			find = metric_dimension.find_pseudo_boolean(P)
	k = UNSET
	k_time : Union[int, timer] = UNSET
	try:
		with timeout(limit):
			with timer() as k_time:
				k = find.minimum()
	except timeout_exception:
		k_time = TIMEOUT
	except:
		k_time = ERROR
	if edge:
		datum['edge_metric_dimension'] = k
		datum['edge_metric_dimension_time'] = k_time
	else:
		datum['metric_dimension'] = k
		datum['metric_dimension_time'] = k_time
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	result = json.dumps(datum, default=float)
	return result

option_spec = [
	(['-a', '--algorithm'], '', None),
	(['--edge'], False, parse_switch),
	(['--timeout'], -1, float)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	method, _, _ = cast(Tuple[str, bool, float], option)
	if method not in ('bruteforce', 'boolean_satisfiability', 'linear_integer_arithmetic', 'pseudo_boolean'):
		return False
	return True

def option_augment(option : Tuple[Any, ...]) -> Tuple[Any, ...]:
	mode, method, limit = cast(Tuple[str, bool, float], option)
	if limit < 0:
		if 'TIMEOUT' in os.environ:
			try:
				limit = float(os.environ['TIMEOUT'])
			except:
				pass
		if limit < 0:
			limit = 0
	option = (mode, method, limit)
	return option

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		usage:
			... -a=<method>>

		options:
			-a=<method>, --algorithm=<method>
				bruteforce
				boolean_satisfiability
				linear_integer_arithmetic
				pseudo_boolean

			--edge
				Find edge metric dimension instead of metric dimension.

			--timeout=<seconds>
				Stop search when over the specified amount of time.
				Defaults to 0, meaning no limit.
	'''
	).strip()
