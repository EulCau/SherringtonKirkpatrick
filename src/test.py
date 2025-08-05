import importlib
import os.path

import numpy as np

import data_manager as dm


def test_method():
	data_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"))
	data_name = "generated_data.pkl"
	data = dm.load_generated_data(data_path, data_name)
	if data is None:
		print("No data loaded, regenerating")
		data = dm.data_generator(data_path, data_name, 30, 10000)

	methods = [
		"simulated_annealing",
		"sdp_relaxation",
	]

	accuracy = np.zeros(len(methods))
	for J_flat, _, best_E in data:
		J = dm.flat2J(J_flat)
		for i in range(len(methods)):
			method_name = methods[i]
			try:
				module = importlib.import_module(f"methods.{method_name}")
				compute_E = module.compute_min_energy(J)[1]
				if np.isclose(compute_E, best_E, atol=1e-5):
					accuracy[i] += 1
			except ImportError:
				print(f"error: can't find {method_name}")
			except AttributeError:
				print(f"error: can't find compute_min_energy for {method_name}")

	for i in range(len(methods)):
		print(f"{methods[i]}: {accuracy[i] / len(data)}")


if __name__ == "__main__":
	test_method()
