from time import perf_counter
import signal
from typing import Any, Literal, Optional, Type, TypeVar

T = TypeVar('T')

class timer():
	def __enter__(self) -> 'timer':
		self.start = perf_counter()
		self.elapsed = 0.0
		return self

	def __exit__(self, *args : object, **kwargs : object) -> None:
		self.end = perf_counter()
		self.elapsed = self.end - self.start

	def __float__(self) -> float:
		return self.elapsed

	def __repr__(self) -> str:
		return str(float(self))

class timeout_exception(Exception):
	pass

class timeout:
	def __init__(self, seconds: float) -> None:
		self.seconds = seconds
		self._old_handler : Any = None

	def _handle_timeout(self, signum : int, frame : Any) -> None:
		raise timeout_exception()

	def __enter__(self) -> 'timeout':
		self._old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
		signal.setitimer(signal.ITIMER_REAL, self.seconds)
		return self

	def __exit__(self, exc_type : Optional[Type[BaseException]], exc_value : Optional[BaseException], traceback : Any) -> Literal[False]:
		signal.setitimer(signal.ITIMER_REAL, 0)
		if self._old_handler is not None:
			signal.signal(signal.SIGALRM, self._old_handler)
		return False
