#include <fstream>
#include <iostream>
#include <vector>
#include <cfloat>
#include "solver.cuh"

std::vector<unsigned int> find_min_configs(const float* energies, const size_t total_k) {
    float min_energy = FLT_MAX;
    for (size_t i = 0; i < total_k; ++i) {
        if (energies[i] < min_energy)
            min_energy = energies[i];
    }

    std::vector<unsigned int> result;
    for (size_t i = 0; i < total_k; ++i) {
        if (energies[i] == min_energy)
            result.push_back(i);
    }
    return result;
}

int main() {
    constexpr int N = 30;
    std::vector J(N * N, 0.0f);

    std::ifstream fin("../data/sk30Jij.txt");
    if (!fin.is_open()) {
        std::cerr << "Error: Failed to open Jij file.\n";
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        if (!(fin >> J[i])) {
            std::cerr << "Error: Failed to read enough data from file.\n";
            return 1;
        }
    }

    fin.close();

    std::vector<int> best_S;
    float* energies = nullptr;

    launch_energy_kernel(J.data(), N, energies);
    std::vector<unsigned int> result = find_min_configs(energies, 1ULL << N);

    for (unsigned int i : result)
    {
        for (size_t j = 0; j < N; ++j)
        {
            std::cout << (((i >> j) & 1) ? 1 : -1) << ' ';;
        }
        std::cout << std::endl;
    }

    std::cout << energies[result[0]] << std::endl;

    return 0;
}
