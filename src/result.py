import importlib
import os.path

import numpy as np

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
data_name="sk30Jij.npy"
data_slice=0
run_exhaustive_search = False

methods = [
	("exhaustive_search", None),
	("simulated_annealing", 1),
	("sdp_relaxation", 42),
]


best_S_with_given_J = np.array(
	[-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1])
best_E_with_given_J = -21.62167
information = "Warning: The exhaustive search did not execute; "\
			  "the results are based solely on the initially provided first set from the given sets."

data = np.load(os.path.join(data_path, data_name), allow_pickle=True)
J =  data[data_slice]

for method_name, seed in methods:
	if (method_name == "exhaustive_search") and not run_exhaustive_search:
		print(method_name, information, best_S_with_given_J, best_E_with_given_J, sep="\n", end="\n\n")
		continue
	try:
		module = importlib.import_module(f"methods.{method_name}")
		best_S, best_E = module.compute_min_energy(J, seed)
		print(method_name, best_S, best_E, sep="\n", end="\n\n")
	except ImportError:
		print(f"error: can't find {method_name}")
	except AttributeError:
		print(f"error: can't find compute_min_energy for {method_name}")
