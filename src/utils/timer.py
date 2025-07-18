from time import perf_counter
from typing import Any, TypeVar

T = TypeVar('T')

class timer():
	def __enter__(self) -> 'timer':
		self.start = perf_counter()
		self.elapsed = 0.0
		return self

	def __exit__(self, *_ : Any) -> None:
		self.end = perf_counter()
		self.elapsed = self.end - self.start

	def __float__(self) -> float:
		return self.elapsed

	def __repr__(self) -> str:
		return str(float(self))
