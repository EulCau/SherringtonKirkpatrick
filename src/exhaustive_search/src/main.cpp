#include <fstream>
#include <iostream>
#include <vector>
#include <cfloat>
#include <unistd.h>
#include <climits>
#include <string>
#include "solver.cuh"

std::string get_executable_dir() {
    char path[PATH_MAX];
    const ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    if (count == -1) {
        throw std::runtime_error("Failed to read /proc/self/exe");
    }

    const std::string full_path(path, count);
    const size_t last_slash = full_path.find_last_of('/');
    return full_path.substr(0, last_slash);
}

unsigned int find_min_configs(const float* energies, const size_t total_k) {
    float min_energy = FLT_MAX;
    unsigned int result = 0;

    for (size_t i = 0; i < total_k; ++i) {
        if (energies[i] < min_energy)
        {
            min_energy = energies[i];
            result = i;
        }
    }

    return result;
}

int main() {
    constexpr int N = 30;
    std::vector J(N * N, 0.0f);

    std::string exec_dir = get_executable_dir();
    std::string jij_path = exec_dir + "/../data/sk30Jij.txt";
    std::ifstream fin(jij_path);

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
    unsigned int result = find_min_configs(energies, 1ULL << N);

    for (size_t j = 0; j < N; ++j)
    {
        std::cout << (((result >> j) & 1) ? 1 : -1) << ' ';;
    }
    std::cout << std::endl;

    std::cout << energies[result] << std::endl;

    return 0;
}
