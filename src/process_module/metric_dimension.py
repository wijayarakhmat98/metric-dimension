import json
import metric_dimension
import re
from typing import Any, cast, Dict, Optional, Tuple
from utils import timer

preserve_order = False
header = None

def decode(result : str) -> Optional[Dict[str, Any]]:
	try:
		datum = cast(Dict[str, Any], json.loads(result))
		return datum
	except:
		return None

def transform(datum : Optional[Dict[str, Any]], option : Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
	if not datum:
		return None
	mode, method = cast(Tuple[str, str], option)
	if mode not in ('metric-dimension', 'edge-metric-dimension'):
		return None
	if method not in ('brute-force', 'boolean-satisfiability', 'linear-integer-arithmetic', 'pseudo-boolean'):
		return None
	s = datum['graph']
	M = metric_dimension.graph6_decode(s)
	DV = metric_dimension.distance_matrix(M)
	match mode:
		case 'metric-dimension':
			D = DV
		case 'edge-metric-dimension':
			E = metric_dimension.edges(M)
			DE = metric_dimension.edge_distance_matrix(E, DV)
			D = DE
	B = metric_dimension.distance_similarity(D)
	P = metric_dimension.reduced_distance_similarity(B)
	match method:
		case 'brute-force':
			find = metric_dimension.find_bruteforce
		case 'boolean-satisfiability':
			find = metric_dimension.find_boolean_satisfiability
		case 'linear-integer-arithmetic':
			find = metric_dimension.find_linear_integer_arithmetic
		case 'pseudo-boolean':
			find = metric_dimension.find_pseudo_boolean
	with timer() as k_time: k = find(P)
	match mode:
		case 'metric-dimension':
			datum['metric_dimension'] = k
			datum['metric_dimension_time'] = k_time
		case 'edge-metric-dimension':
			datum['edge_metric_dimension'] = k
			datum['edge_metric_dimension_time'] = k_time
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	result = json.dumps(datum, default=float)
	return result

option_spec = [
	(['-f', '--find'], '', None),
	(['-a', '--algorithm'], '', None)
]

def option_valid(option : Tuple[Any, ...]) -> bool:
	mode, method = cast(Tuple[str, str], option)
	if mode not in ('metric-dimension', 'edge-metric-dimension'):
		return False
	if method not in ('brute-force', 'boolean-satisfiability', 'linear-integer-arithmetic', 'pseudo-boolean'):
		return False
	return True

option_augment = None

def help() -> str:
	return re.sub(r'\n\t\t\t', r'\n',
	'''
		usage:
			... <-f=<mode> -a=<method>>

		options:
			-f=<mode>, --find=<mode>
				metric-dimension
				edge-metric-dimension

			-a=<method>, --algorithm=<method>
				brute-force
				boolean-satisfiability
				linear-integer-arithmetic
				pseudo-boolean
	'''
	).strip()
