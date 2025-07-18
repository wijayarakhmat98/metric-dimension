from typing import Dict, List, Tuple

def parse_args(args : List[str], config : List[Tuple[List[str], str, str]]) -> Dict[str, str]:
	map : Dict[str, str] = {}
	option : Dict[str, str] = {}
	for flags, key, value in config:
		for flag in flags:
			map[flag] = key
		option[key] = value
	for arg in args:
		key, value = [s.strip() for s in (arg.split('=', 1) + [''])[:2]]
		if key in map:
			option[map[key]] = value
	return option
