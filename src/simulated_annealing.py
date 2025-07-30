import numpy as np

def energy(S, J):
	return -0.5 * S @ J @ S

def generate_neighbor(S):
	S_new = S.copy()
	i = np.random.randint(len(S))
	S_new[i] *= -1  # flip one spin
	return S_new

def compute_min_energies(J, seed=None, T_init=10.0, T_min=1e-3, alpha=0.98, max_iter=10000):
	if seed is not None:
		np.random.seed(seed)
	n = J.shape[0]
	S = np.random.choice([-1, 1], size=n)
	E = energy(S, J)

	T = T_init
	best_S = S.copy()
	best_E = E

	for it in range(max_iter):
		S_new = generate_neighbor(S)
		E_new = energy(S_new, J)
		dE = E_new - E

		if dE < 0 or np.random.rand() < np.exp(-dE / T):
			S = S_new
			E = E_new

		if E < best_E:
			best_S = S.copy()
			best_E = E

		T *= alpha
		if T < T_min:
			break

	return best_S, best_E
