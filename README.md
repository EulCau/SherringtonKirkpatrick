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
* **Semidefinite Programming (SDP) Relaxation**
* And others

The goal is to compare the performance and effectiveness of these algorithms, providing a solid foundation for subsequent theoretical analysis and practical applications.

---

## Algorithm Details

### 1. Exhaustive Search (CUDA-Accelerated, Reference Only)

Due to the relatively small dimensionality of the input data (only 30 spins), it is feasible to perform an exhaustive search over all spin configurations using CUDA. This algorithm has been tested on a Linux system equipped with an NVIDIA 4060 Ti GPU. In theory, any NVIDIA GPU with **more than 4 GiB of memory** should be able to run it without modification.

> **Note:** This method is *not* intended as a general-purpose algorithm. It is included solely to provide a **ground truth reference** for evaluating the accuracy of other algorithms in this project.

#### Core Idea

* First, note that the Hamiltonian satisfies the symmetry

  $$
  H\left(S\right) = H\left(-S\right),
  $$

  which allows us to **fix the last spin to -1**, reducing the search space by half.
  Similarly, for consistency, all later methods **only report one representative** of such symmetric configurations, and this property will not be reiterated.

* To avoid excessive data transfer and expensive configuration generation, each spin configuration is encoded as an unsigned long integer. Its binary representation (from least to most significant bit) corresponds to spins from the first to the last particle, with `0` representing spin $-1$ and `1` representing spin $+1$.

* We compute the energy of all configurations in parallel, store the energies in a global array, and use a CPU reduction to find the minimum.
  The total work is

  $$
  O\left(N^2 \cdot 2^{N-1}\right),
  $$

  while the parallel span is

  $$
  O\left(N^2\right).
  $$

  Although CUDA threads may not fully reduce the time complexity to $O\left(N^2\right)$ due to hardware constraints, thread multiplexing ensures the computation remains within practical bounds. The total expected time is on the order of **10 million operations**, which is well within the processing capability of modern GPUs.

* The memory usage consists of storing the coupling matrix ($N^2$ floats) and the energy array ($2^{N-1}$ floats), totaling roughly **2 GiB** of VRAM. With additional overhead, a **4 GiB GPU** is sufficient.

In practice, this method takes only **1–2 seconds** to find the true ground state configuration.

### 2. Simulated Annealing

Among all the implemented algorithms, Simulated Annealing is arguably the simplest, most classical, and most physically intuitive approach for this problem. It uses a gradually decreasing temperature variable `T` to control the acceptance probability of energy-increasing transitions. This mechanism allows the algorithm to more easily escape local minima in the early stages and to converge toward a local optimum in the later stages. This behavior mirrors the physical intuition that, in the early phase, a system is thermally unstable and more likely to escape from local energy minima due to thermal fluctuations.

#### Core Idea

* The algorithm maintains a temperature $T$ that decays over time. At each iteration, a new state $S^*$ is generated from the current state $S$ by flipping a randomly selected spin (i.e., changing its sign).

* The energy difference is computed as

  $$
  \Delta E = H\left(S^*\right) - H\left(S\right),
  $$

  and the new state is accepted with probability

  $$
  P = \exp\left(-\frac{\Delta E}{T}\right).
  $$

  When $\Delta E < 0$, $P > 1$, so the new state is always accepted.

* The temperature $T$ typically follows an exponential decay schedule, for example:

  $$
  T \leftarrow \alpha T,\quad \alpha < 1.
  $$

  Alternatively, more complex decay schemes can be used.

* In fact, as will be discussed in the **"Other Methods"** section, we plan to explore approaches where a neural network dynamically determines <font color='red'> whether to accept new states and/or how to decay the temperature. </font>

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
