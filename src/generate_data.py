import numpy as np
from sdp_relaxation import compute_min_energy


def generate_one(N=30, seed=None, num_try=3):
	if seed is not None:
		np.random.seed(seed)

	best_S = np.zeros(N).astype(int)
	best_energy = np.inf
	J_plat = np.random.normal(0, 0.2, N * (N-1) // 2)
	J = flat2J(J_plat, N)

	for i in range(num_try):
		S, energy = compute_min_energy(J)
		if energy < best_energy:
			best_energy = energy
			best_S = S

	if best_S[-1] == 1:
		best_S = -best_S

	return J_plat, best_S


def flat2J(J_flat, N=None):
	n_data = len(J_flat)
	if N is None:
		N = int(np.sqrt(n_data * 2) + 0.5)

	J = np.zeros((N, N))
	indices = np.triu_indices(N, k=1)
	J[indices] = J_flat
	return J + J.T


if __name__ == "__main__":
	print(generate_one())
