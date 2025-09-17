import json
import multiprocessing
from typing import Any, Dict, List

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def decode_then_stringify(result : str) -> Dict[str, Any]:
	datum = decode(result)
	for key, value in datum.items():
		datum[key] = str(value)
	return datum

def format(results : List[str], *args : object, **kwargs : object) -> None:
	if not results:
		return
	print_header = True
	with multiprocessing.Pool() as pool:
		for datum in pool.imap(decode_then_stringify, results):
			if print_header:
				print(','.join(datum.keys()))
				print_header = False
			print(','.join(datum.values()))
