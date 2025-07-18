import json
import multiprocessing
from typing import Any, Dict, List

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	time_keys = [key for key in datum if 'time' in key]
	for key in time_keys:
		del datum[key]
	return datum

def format(results : List[str], sort : bool, *_ : object) -> None:
	with multiprocessing.Pool() as pool:
		data = list(pool.imap_unordered(decode, results))
	if sort:
		data.sort(key=lambda datum: datum['graph'])
	for datum in data:
		print(json.dumps(datum))
