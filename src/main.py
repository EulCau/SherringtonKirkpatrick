import importlib

import numpy as np


def load_data(data_path="../data/", data_name="sk30Jij.npy", data_slice=0):
	data = np.load(data_path+data_name, allow_pickle=True)
	return data[data_slice]

def main(run_exhaustive_search):
	methods = [
		"exhaustive_search",
		"simulated_annealing",
		"sdp_relaxation",
	]
	J = load_data()
	best_S_with_no_change_J = np.array(
		[-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1])
	best_E_with_no_change_J = -21.6217
	information = "Warning: The exhaustive search did not execute; "\
				  "the results are based solely on the initially provided first set from the given sets."

	for method_name in methods:
		if (method_name == "exhaustive_search") and not run_exhaustive_search:
			print(method_name, information, best_S_with_no_change_J, best_E_with_no_change_J, sep="\n", end="\n\n")
			continue
		try:
			module = importlib.import_module(method_name)
			best_S, best_E = module.compute_min_energy(J, seed=42)
			print(method_name, best_S, best_E, sep="\n", end="\n\n")
		except ImportError:
			print(f"error: can't find {method_name}")
		except AttributeError:
			print(f"error: can't find compute_min_energy for {method_name}")


if __name__ == "__main__":
	main(False)
