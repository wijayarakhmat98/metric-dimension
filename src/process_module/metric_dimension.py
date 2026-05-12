import json
import metric_dimension
import os
import re
from typing import Any, cast, Dict, List, Optional, Tuple
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
	mode : List[bool] = []
	if 'metric_dimension' in datum and datum['metric_dimension'] is None:
		mode.append(False)
	if 'edge_metric_dimension' in datum and datum['edge_metric_dimension'] is None:
		mode.append(True)
	if not mode:
		return datum
	if 'algorithm' in datum:
		method = datum['algorithm']
		if method not in metric_dimension.ALGORITHMS:
			return datum
	else:
		return datum
	limit, = cast(Tuple[float], option)
	for edge in mode:
		with timer() as total_time:
			s = datum['graph']
			M = metric_dimension.graph6_decode(s)
			DV = metric_dimension.distance_matrix(M)
			E = metric_dimension.edges(M)
			DE = metric_dimension.edge_distance_matrix(E, DV)
			if edge:
				D = DE
			else:
				D = DV
			B = metric_dimension.distance_similarity(D)
			P = metric_dimension.reduced_distance_similarity(B)
			find = metric_dimension.create_find(P, method, limit)
			k, k_time, internal_time = find.minimum()
		datum['vertices'] = M.shape[0]
		datum['edges'] = E.shape[0]
		if edge:
			datum['edge_metric_dimension'] = k
			datum['edge_metric_dimension_time'] = k_time
			datum['edge_metric_dimension_internal_time'] = internal_time
			datum['edge_metric_dimension_total_time'] = total_time
		else:
			datum['metric_dimension'] = k
			datum['metric_dimension_time'] = k_time
			datum['metric_dimension_internal_time'] = internal_time
			datum['metric_dimension_total_time'] = total_time
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	result = json.dumps(datum, default=float)
	return result

option_spec = [
	(['--timeout'], -1, float)
]

option_valid = None

def option_augment(option : Tuple[Any, ...]) -> Tuple[Any, ...]:
	limit, = cast(Tuple[float], option)
	if limit < 0:
		if 'TIMEOUT' in os.environ:
			try:
				limit = float(os.environ['TIMEOUT'])
			except:
				pass
		if limit < 0:
			limit = 0
	option = (limit,)
	return option

def help() -> str:
	return re.sub(r'\n\t\t', r'\n',
	'''
		usage:
			... [--timeout=<seconds>]

		options:
			--timeout=<seconds>
				Stop search when over the specified amount of time.
				Defaults to 0, meaning no limit.
	'''
	).strip()
