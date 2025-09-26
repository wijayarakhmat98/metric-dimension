from typing import Iterator, Self, TextIO

class read_file(Iterator[str]):
	def __init__(self, file : TextIO):
		self.file = file

	def __iter__(self) -> Self:
		return self

	def __next__(self) -> str:
		for raw in self.file:
			line = raw.strip()
			if not line:
				continue
			return line
		raise StopIteration
