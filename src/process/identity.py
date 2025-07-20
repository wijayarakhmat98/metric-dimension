import json
from typing import Any, Dict

def process(graph : str, *args : object, **kwargs : object) -> str:
	return json.dumps({
		'graph': graph
	}, default=float)

def resume(result : str, *args : object, **kwargs : object) -> str:
	datum : Dict[str, Any] = json.loads(result)
	graph : str = datum['graph']
	return graph
