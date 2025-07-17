import json
from typing import Any, Dict, List

def decode(result : str) -> Dict[str, Any]:
	datum : Dict[str, Any] = json.loads(result)
	return datum

def format(data : List[Dict[str, Any]]) -> None:
	data.sort(key=lambda datum: datum['graph'])
	for datum in data:
		print(json.dumps(datum, indent=2))
