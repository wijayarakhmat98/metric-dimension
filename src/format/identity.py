import json
import multiprocessing
from typing import Any, Dict, Iterator

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def format(results : Iterator[str], *args : object, **kwargs : object) -> None:
	with multiprocessing.Pool() as pool:
		for datum in pool.imap_unordered(decode, results):
			print(json.dumps(datum))
	# if sort:
	# 	data.sort(key=lambda datum: datum['graph'])
