import numpy as np


def exhaustive_sample(J, seed=None, num_samples=10**7):
	if seed is not None:
		np.random.seed(seed)
	n = J.shape[0]
	best_E = float('inf')
	best_S = None
	for i in range(num_samples):
		S = np.random.choice([-1, 1], size=n)
		E = -0.5 * S @ J @ S
		if E < best_E:
			best_E = E
			best_S = S.copy()
		if (i+1)% 10000 == 0:
			print(i+1, E)
	return best_S, best_E
