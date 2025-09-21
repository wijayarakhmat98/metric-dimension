import json

def process(graph : str, *args : object, **kwargs : object) -> str:
	return json.dumps({
		'graph': graph
	}, default=float)
