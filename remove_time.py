import json
from typing import Any, Dict, List

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def format(data : List[Dict[str, Any]]) -> None:
	data.sort(key=lambda datum: datum['graph'])
	for datum in data:
		time_keys = [key for key in datum if 'time' in key]
		for key in time_keys:
			del datum[key]
	for datum in data:
		print(json.dumps(datum))
