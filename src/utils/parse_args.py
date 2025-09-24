from typing import Any, Callable, Dict, List, Optional, Tuple

def parse_args(args : List[str], config : List[Tuple[List[str], Any, Optional[Callable[[str], Any]]]]) -> Tuple[Any, ...]:
	map : Dict[str, int] = {}
	option_list : List[Any] = []
	transform_list : List[Optional[Callable[[str], Any]]] = []
	for i, (flags, value, transform) in enumerate(config):
		for flag in flags:
			map[flag] = i
		option_list.append(value)
		transform_list.append(transform)
	for arg in args:
		flag, value = [s.strip() for s in (arg.split('=', 1) + [''])[:2]]
		if flag not in map:
			continue
		i = map[flag]
		transform = transform_list[i]
		if transform:
			option_list[i] = transform(value)
		else:
			option_list[i] = value
	option = tuple(option_list)
	return option

def parse_switch(s : str) -> bool:
	return not s
