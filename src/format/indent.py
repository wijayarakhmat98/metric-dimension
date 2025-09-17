import json
import multiprocessing
from typing import Any, Dict, List

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def format(results : List[str], *args : object, **kwargs : object) -> None:
	with multiprocessing.Pool() as pool:
		data = list(pool.imap_unordered(decode, results))
	for datum in data:
		print(json.dumps(datum, indent=2))
