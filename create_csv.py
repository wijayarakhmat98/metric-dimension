import json
import multiprocessing
from typing import Any, Dict, List

def format_time(time : float) -> str:
	return '{}'.format(round(1000 * time, 3))

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	for key, value in datum.items():
		if "time" in key:
			datum[key] = format_time(value)
	return datum

def format(results : List[str], *_ : object) -> None:
	with multiprocessing.Pool() as pool:
		data = list(pool.imap_unordered(decode, results))
	data.sort(key=lambda datum: datum['graph'])
	print('graph,vs,b,b_time,ws,ws_time,es,be,be_time,wes,wes_time,t_time')
	for datum in data:
		print('{},{},{},{},{},{},{},{},{},{},{},{}'.format(
			datum['graph'],
			datum['vertices'],
			datum['metric_dimension'],
			datum['metric_dimension_time'],
			datum['metric_dimension_solutions'],
			datum['metric_dimension_solutions_time'],
			datum['edges'],
			datum['edge_dimension'],
			datum['edge_dimension_time'],
			datum['edge_dimension_solutions'],
			datum['edge_dimension_solutions_time'],
			datum['total_time']
		))
