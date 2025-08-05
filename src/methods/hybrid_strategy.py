import os

import numpy as np
from . import sdp_relaxation as sr
from . import simulated_annealing as sa


_num_rounds = 500
_T_init = 0.5


def compute_min_energy(J, seed=None, num_rounds=_num_rounds, **kwargs):
	if seed is not None:
		np.random.seed(seed)

	S_init, _ = sr.compute_min_energy(J, num_rounds=num_rounds)
	return sa.compute_min_energy(J, S_init=S_init, T_init=_T_init, **kwargs)


if __name__ == "__main__":
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "../../data")
	data_name="sk30Jij.npy"
	data_slice=0

	data = np.load(os.path.join(data_path, data_name), allow_pickle=True)
	_J =  data[data_slice]

	_best_S, _best_E = compute_min_energy(_J)

	print(_best_S, _best_E, sep="\n")
