import warnings
import numpy as np
import cvxpy as cp


num_rounds = 100


def solve_sdp(J, seed=None):
	if seed is not None:
		np.random.seed(seed)

	N = J.shape[0]

	X = cp.Variable((N, N), symmetric=True)
	constraints = [X >> 0]
	constraints += [X[i, i] == 1 for i in range(N)]

	objective = cp.Maximize(cp.sum(cp.multiply(J, X)))
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.SCS, verbose=False)

	if X.value is None:
		raise RuntimeError("SDP solver did not find a solution")

	try:
		V = np.linalg.cholesky(X.value + 1e-6 * np.eye(N))
	except np.linalg.LinAlgError:
		eig_vals, eig_vecs = np.linalg.eigh(X.value)
		eig_vals = np.maximum(eig_vals, 0)
		V = np.diag(np.sqrt(eig_vals)) @ eig_vecs.T

	best_E = np.inf
	best_S = None

	for _ in range(num_rounds):
		r = np.random.randn(V.shape[0])
		r /= np.linalg.norm(r)

		projections = V.T @ r
		S = np.sign(projections)

		S[S == 0] = 1

		energy = -S @ J @ S / 2

		if energy < best_E:
			best_E = energy
			best_S = S

	return best_S, best_E


def compute_min_energy(J, seed=None):
	n=J.shape[0]
	if n <= 100:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return solve_sdp(J, seed)
	else:
		return solve_sdp(J, seed)
