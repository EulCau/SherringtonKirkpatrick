import importlib
import os.path
import time

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
		"hybrid_strategy",
	]

	accuracy = np.zeros(len(methods))
	time_takes = np.zeros(len(methods))
	for i in range(len(methods)):
		start = time.perf_counter()
		for J_flat, _, best_E in data:
			J = dm.flat2J(J_flat)
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
		end = time.perf_counter()
		time_takes[i] = end - start

	for i in range(len(methods)):
		print(f"{methods[i]}: {accuracy[i] / len(data)}; {time_takes[i] / len(data) * 1000:.3f}ms")


if __name__ == "__main__":
	seed = 42
	np.random.seed(seed)
	test_method()
