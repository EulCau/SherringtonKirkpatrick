import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import subprocess
import os

import numpy as np

solver_path = os.path.join(os.path.dirname(__file__), "..", "exhaustive_search", "bin", "solver")

script_dir = os.path.dirname(os.path.abspath(__file__))
cu_path = os.path.join(script_dir, "..", "exhaustive_search", "src")
include_path = os.path.join(script_dir, "..", "exhaustive_search", "include")
with open(os.path.join(cu_path, "solver.cu"), "r") as f:
	kernel_code = f.read()
	mod = SourceModule(
		kernel_code,
		options=[f"-I{include_path}"],
		nvcc="/usr/local/cuda-12.8/bin/nvcc"
	)

compute_energies_and_min = mod.get_function("compute_energies_and_min")
reduce_min_energy = mod.get_function("reduce_min_energy")
final_reduce_min = mod.get_function("final_reduce_min")


def compute_min_energy(J, seed=None):
	_ = seed

	idx = find_min_idx(J)
	best_S, best_E = idx2result(J, idx)

	return best_S, best_E


def find_min_idx(J):
	assert J.ndim == 2 and J.shape[0] == J.shape[1], "S must be square matrix"
	assert J.dtype == np.float32, "S must be  in type float32"

	N = J.shape[0]
	total_k = 1 << (N - 1)

	d_J = cuda.mem_alloc(J.nbytes)
	cuda.memcpy_htod(d_J, J)

	d_energies = cuda.mem_alloc(total_k * np.float32().nbytes)
	d_min_index = cuda.mem_alloc(np.uint32().nbytes)

	threads_per_block = 256
	energy_blocks = (total_k + threads_per_block - 1) // threads_per_block

	compute_energies_and_min(
		d_J, np.int32(N), d_energies,
		block=(threads_per_block, 1, 1),
		grid=(energy_blocks, 1)
	)
	pycuda.autoinit.context.synchronize()

	reduce_blocks = (total_k + threads_per_block * 2 - 1) // (threads_per_block * 2)

	d_block_min_energy = cuda.mem_alloc(reduce_blocks * np.float32().nbytes)
	d_block_min_index = cuda.mem_alloc(reduce_blocks * np.uint32().nbytes)

	reduce_min_energy(
		d_energies, np.uint32(total_k),
		d_block_min_energy, d_block_min_index,
		block=(threads_per_block, 1, 1),
		grid=(reduce_blocks, 1),
		shared=threads_per_block * np.float32().nbytes
	)
	pycuda.autoinit.context.synchronize()

	final_reduce_min(
		d_block_min_energy, d_block_min_index, np.int32(reduce_blocks), d_min_index,
		block=(1, 1, 1), grid=(1, 1)
	)
	pycuda.autoinit.context.synchronize()

	h_min_index = np.zeros(1, dtype=np.uint32)
	cuda.memcpy_dtoh(h_min_index, d_min_index)

	d_J.free()
	d_energies.free()
	d_min_index.free()
	d_block_min_energy.free()
	d_block_min_index.free()

	return int(h_min_index[0])


def idx2result(J, idx):
	N = J.shape[0]
	best_S = np.zeros(N, dtype=J.dtype)

	for i in range(N):
		best_S[i] = 1 if (idx >> i & 1) else -1

	best_E = - best_S @ J @ best_S / 2

	return best_S.astype(int), best_E


if __name__ == "__main__":
	data_path = "../../data"
	data_name="sk30Jij.npy"
	data_slice=0

	data = np.load(os.path.join(data_path, data_name), allow_pickle=True)
	J =  data[data_slice]

	_best_S, _best_E = compute_min_energy(J)

	print(_best_S, _best_E, sep="\n")
