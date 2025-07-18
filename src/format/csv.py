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
		datum[key] = str(value)
	return datum

def format(results : List[str], sort : bool, *_ : object) -> None:
	if not results:
		return
	with multiprocessing.Pool() as pool:
		data = list(pool.imap_unordered(decode, results))
	if sort:
		data.sort(key=lambda datum: datum['graph'])
	print(','.join(data[0].keys()))
	for datum in data:
		print(','.join(datum.values()))
