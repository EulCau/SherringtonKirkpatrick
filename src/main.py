import numpy as np

from simulated_annealing import simulated_annealing

def load_data(data_path="../data", data_name="sk30Jij.npy", data_slice=0):
	data = np.load(data_path+data_name, allow_pickle=True)
	return data[data_slice]

def main():
	data_path = "../data/"
	data_name = "sk30Jij.npy"
	data_slice = 0
	J = load_data(data_path, data_name, data_slice)
	best_S, best_E = simulated_annealing(J, 1)
	print(best_S, best_E)

if __name__ == "__main__":
	main()
