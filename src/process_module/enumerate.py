import json
import itertools
import metric_dimension
import numpy as np
import numpy.typing as npt
from typing import Any, cast, Dict, List, Optional, Tuple

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
	if 'metric_dimension' in datum and datum['metric_dimension'] is not None:
		mode.append(False)
	if 'edge_metric_dimension' in datum and datum['edge_metric_dimension'] is not None:
		mode.append(True)
	if not mode:
		return datum
	for edge in mode:
		s = datum['graph']
		M = metric_dimension.graph6_decode(s)
		DV = metric_dimension.distance_matrix(M)
		E = metric_dimension.edges(M)
		if edge:
			DE = metric_dimension.edge_distance_matrix(E, DV)
			D = DE
			k = datum['edge_metric_dimension']
		else:
			D = DV
			k = datum['metric_dimension']
		B = metric_dimension.distance_similarity(D)
		P = metric_dimension.reduced_distance_similarity(B)
		Xs = find_enumerate(P, k)
		if edge:
			datum['edge_enumerate'] = Xs
		else:
			datum['enumerate'] = Xs
	return datum

def encode(datum : Optional[Dict[str, Any]]) -> Optional[str]:
	if not datum:
		return None
	return json.dumps(datum)

def find_enumerate(P : npt.NDArray[np.bool], k : int) -> List[List[str]]:
	Ws : List[npt.NDArray[np.bool_]] = []
	nV = P.shape[0]
	_P = P[:, P.sum(axis=0) >= k]
	combinations = itertools.combinations(range(nV), k)
	for indices in combinations:
		W = np.zeros((nV, 1), dtype=np.bool_)
		W[indices, 0] = True
		is_subset = np.all((W | _P) == _P, axis=0)
		if not np.any(is_subset):
			Ws.append(W)
	X = np.array(['x{}'.format(i) for i in range(1, nV + 1)]).reshape(-1, 1)
	Xs = [X[W].tolist() for W in Ws]
	return Xs

option_spec = None
option_valid = None
option_augment = None

def help() -> str:
	return ''
