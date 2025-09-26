import importlib
from types import ModuleType

def load_module(module_name : str) -> ModuleType:
	module = importlib.import_module(module_name)
	return module
