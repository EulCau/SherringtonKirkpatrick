import subprocess
import os

import numpy as np

solver_path = os.path.join(os.path.dirname(__file__), "exhaustive_search", "bin", "solver")


def compute_min_energy(_=None, seed=None):
	__=seed
	try:
		result = subprocess.run(
			[solver_path],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			check=True,
			universal_newlines=True
		)
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Solver failed:\n{e.stderr}")

	output_lines = result.stdout.strip().splitlines()

	if len(output_lines) < 2:
		raise ValueError(f"Unexpected solver output:\n{result.stdout}")

	best_S = np.array(list(map(int, output_lines[0].strip().split())))
	best_E = float(output_lines[1].strip())

	return best_S, best_E


if __name__ == "__main__":
	S, E = compute_min_energy()
	print("Best Energy:", E)
	print("Best Configuration:", S)
