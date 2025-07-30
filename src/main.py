import importlib

import numpy as np


def load_data(data_path="../data/", data_name="sk30Jij.npy", data_slice=0):
	data = np.load(data_path+data_name, allow_pickle=True)
	return data[data_slice]

def main():
	data_path = "../data/"
	data_name = "sk30Jij.npy"
	data_slice = 0
	methods = [
		"simulated_annealing",
	]
	J = load_data(data_path, data_name, data_slice)

	for method_name in methods:
		try:
			module = importlib.import_module(method_name)
			best_S, best_E = module.compute_min_energies(J, data_slice)
			print(method_name, best_S, best_E, sep="\n", end="\n\n")
		except ImportError:
			print(f"error: can't find {method_name}")
		except AttributeError:
			print(f"error: can't find compute_min_energies for {method_name}")


if __name__ == "__main__":
	main()
