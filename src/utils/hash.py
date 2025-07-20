import hashlib

def hash(s : str) -> str:
	b = s.encode()
	_h = hashlib.sha256(b)
	h = _h.hexdigest()
	return h
