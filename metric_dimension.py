import numpy as np
import numpy.typing as npt
from typing import Any, cast, List
import z3 # type: ignore

def variable(n : int) -> npt.NDArray[Any]:
	_vs = [z3.Int('x{}'.format(i + 1)) for i in range(n)] # type: ignore
	vs = np.array(_vs)
	return vs

def distance_similarity(d : npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
	p : npt.NDArray[np.bool_] = d[None, :, :] == d[:, None, :]
	p = p.reshape(-1, p.shape[2])
	p = np.unique(p, axis=0)
	p = p[np.any(p, axis=1) & ~np.all(p, axis=1)]
	return p

def find(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_]) -> int:
	z = z3.Optimize()
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # type: ignore
	k = z3.Int('k') # type: ignore
	z.add(z3.Sum(*vs) == k) # type: ignore
	z.add(k >= 1) # type: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < k for c in p])) # type: ignore
	z.minimize(k) # type: ignore
	z.check() # type: ignore
	b = cast(int, z.model()[k].as_long()) # type: ignore
	return b

def enumerate(vs : npt.NDArray[Any], p : npt.NDArray[np.bool_], n : int) -> npt.NDArray[np.bool_]:
	z = z3.Solver()
	z.add(z3.And([z3.And(v >= 0, v <= 1) for v in vs])) # type: ignore
	z.add(z3.Sum(*vs) == n) # type: ignore
	z.add(z3.And([z3.Sum(*vs[c]) < n for c in p])) # type: ignore
	ws : List[List[int]] = []
	while z.check() == z3.sat: # type: ignore
		model = z.model()
		w = [cast(int, model[v].as_long()) for v in vs] # type: ignore
		z.add(z3.Or([v != v_ for v, v_ in zip(vs, w)])) # type: ignore
		ws.append(w)
	return np.array(ws, dtype=np.bool_)

def resolving_representation(w : npt.NDArray[np.bool_], d : npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
	return d[:, w]

def is_resolving_valid(r : npt.NDArray[np.int_]) -> bool:
	r_ = np.unique(r, axis=0)
	valid = r.shape[0] == r_.shape[0]
	return valid
