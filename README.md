# Sherrington-Kirkpatrick model

This project focuses on solving the ground state energy problem of the classical **Sherrington-Kirkpatrick (SK) model**. The SK model is a fundamental example of spin glass systems in statistical physics, defined by a set of interacting spin variables $S = \left(S_{1}, S_{2}, \dots, S_{N}\right)$, where each spin $S_{i}$ takes values $\pm 1$.

The Hamiltonian (energy function) of the model is given by:

$$
H(S) = - \sum_{1 \leq i < j \leq N} J_{ij} S_{i} S_{j}
$$

Here, the coupling matrix $J = \left(J_{ij}\right) \in \mathbb{R}^{N \times N}$ represents the interaction strengths between spins. The goal is to find the spin configuration $S$ that minimizes the system energy $H\left(S\right)$.

---

## Mathematical Abstraction

Assuming $J$ is a symmetric matrix with zero diagonal entries, the Hamiltonian can be expressed as

$$
H\left(S\right) = -\frac{1}{2} S^{T} J S.
$$

The problem then reduces to finding

$$
S^* = \arg \min_{S \in \left\{\pm 1\right\}^N} H\left(S\right).
$$

This problem holds significant importance in both combinatorial optimization and computational physics. It is a typical **Quadratic Unconstrained Binary Optimization (QUBO)** problem, known to be NP-hard, with computational complexity growing exponentially as the number of spins increases.

---

## Algorithm Selection

Traditional methods struggle to directly solve large-scale instances of the problem. Therefore, this project implements multiple optimization algorithms, including:

* **Exhaustive Search** (CUDA-accelerated, used solely to identify the true minimum for benchmarking purposes)
* **Simulated Annealing**
* **Genetic Algorithm**
* **Semidefinite Programming (SDP) Relaxation**
* And others

The goal is to compare the performance and effectiveness of these algorithms, providing a solid foundation for subsequent theoretical analysis and practical applications.

---

## Algorithm Details

blabla

---

## Project Structure

```text
SherringtonKirkpatrick
├── README.md
├── .gitignore
├── data
│   └── sk30Jij.npy
└── src
    ├── npy2txt.py
    ├── main.py
    ├── exhaustive_search.py
    ├── simulated_annealing.py
    ├── ...
    ├── ...
    ├── ...
    └── exhaustive_search
        ├── CMakeLists.txt
        ├── data
        │   └── sk30Jij.txt
        ├── include
        │   └── solver.cuh
        └── src
            ├── main.cpp
            └── solver.cu
```

---

## Usage Instructions

If you wish to use the first set from the provided three sets of $J_{ij}$ (a NumPy array of shape `(3, 30, 30)`) as input data, you don't need to manually change any files.

If you want to switch to a different dataset **without** using exhaustive search, you should modify `main.loaddata` accordingly to correctly load your data.

If you intend to use exhaustive search with a new dataset to find the true ground state energy, please follow these steps:

1. **Ensure your system meets the requirements**:

    * NVIDIA GPU
    * At least $\left(2^{N-29}+2\right)$ GiB of GPU memory

2. **Configure the build system**:

    * Modify `exhaustive_search/CMakeLists.txt` or set appropriate environment variables to match your system configuration.

3. **Prepare the data**:

    * Edit and run `npy2txt.py` to convert your `.npy` dataset to the correct input format.
    * In `main.py`, set the argument to `main()` as `True` to enable exhaustive search mode.
    * Update the data path in `exhaustive_search/src/main.cpp` to point to your converted file.
      Use an **absolute path**, or ensure the program correctly resolves the executable path for a relative path.

4. **Build the executable**:
   Run the following commands inside the `exhaustive_search/` directory:

   ```shell
   mkdir build
   cd build
   cmake ..
   make
   ```

5. **Finalize integration**:

    * After successful compilation, set the global variable `solver_path` in `exhaustive_search.py` to the path of the compiled executable.

Finally, run `main.py` to perform the computation.
