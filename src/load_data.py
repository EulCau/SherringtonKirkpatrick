import os
import pickle

import numpy as np

from generate_data import generate_one

def load_data(N=30, num=10000, val_ratio=0.2, regenerate=False):
	if regenerate or (data := load_saved_data()) is None or len(data) != num:
		print("data generating...")
		data = [None] * num
		for i in range(num):
			data[i] = (generate_one(N))
		print("data generated")

	all_indices = range(num)
	sample_size = int(num * val_ratio)
	val_indices = np.random.choice(all_indices, size=sample_size, replace=False)
	val_indices.sort()

	return data, val_indices


def save_data(data, filename="../data/sk_data.pkl"):
	with open(filename, "wb") as f:
		pickle.dump(data, f)  # type: ignore[arg-type]
	print(f"data saved to {filename}")


def load_saved_data(filename="../data/sk_data.pkl"):
	if os.path.exists(filename):
		with open(filename, "rb") as f:
			return pickle.load(f)
	return None


if __name__ == "__main__":
	_data, _ = load_data(regenerate=True)
	save_data(_data)
