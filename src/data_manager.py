import os
import pickle

import numpy as np

from methods import exhaustive_search as es

_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
def data_generator(data_path=_data_path, data_name="generated_data.pkl", N=30, num = 10000):
	data = []

	print("Starting data generation ...")
	for i in range(num):
		if (i + 1) ==num:
			print(f"  generating data {i+1}.")
		elif (i + 1)%10==0:
			print(f"  generating data {i+1};")
		data.append(generate_one(N))

	print(f"Finished generating data, saving to {data_path}/{data_name}")
	save_data(data, data_path, data_name)
	print(f"Finished saving to {data_path}/{data_name}")

	return data


def generate_one(N=30):
	J_flat = np.random.normal(0, 0.2, N * (N-1) // 2).astype(np.float32)
	best_S, best_E = es.compute_min_energy(flat2J(J_flat, N))
	return J_flat, best_S, best_E


def load_generated_data(data_path=_data_path, data_name="generated_data.pkl"):
	filename = os.path.join(data_path, data_name)
	if os.path.exists(filename):
		with open(filename, "rb") as f:
			return pickle.load(f)
	return None


def save_data(data, data_path=_data_path, data_name="generated_data.pkl"):
	filename = os.path.join(data_path, data_name)
	with open(filename, "wb") as f:
		pickle.dump(data, f)  # type: ignore[arg-type]
	print(f"data saved to {filename}")


def flat2J(J_flat, N=None):
	n_data = len(J_flat)
	if N is None:
		N = int(np.sqrt(n_data * 2) + 1)

	J = np.zeros((N, N), dtype=J_flat.dtype)
	indices = np.triu_indices(N, k=1)
	J[indices] = J_flat
	return J + J.T


if __name__ == "__main__":
	data_generator()
