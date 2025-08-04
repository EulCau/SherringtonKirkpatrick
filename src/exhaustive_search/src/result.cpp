#include <fstream>
#include <iostream>
#include <vector>
#include <cfloat>
#include <unistd.h>
#include <climits>
#include <string>
#include "solver.cuh"

std::string get_executable_dir()
{
    char path[PATH_MAX];
    const ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    if (count == -1)
    {
        throw std::runtime_error("Failed to read /proc/self/exe");
    }

    const std::string full_path(path, count);
    const size_t last_slash = full_path.find_last_of('/');
    return full_path.substr(0, last_slash);
}

float compute_energy_cpu(const float* J, int N, unsigned int k)
{
    float energy = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        int Si = ((k >> i) & 1) ? 1 : -1;
        for (int j = i + 1; j < N; ++j)
        {
            int Sj = ((k >> j) & 1) ? 1 : -1;
            energy -= J[i * N + j] * Si * Sj;
        }
    }
    return energy;
}

int main()
{
    constexpr int N = 30;
    std::vector J(N * N, 0.0f);

    std::string exec_dir = get_executable_dir();
    std::string jij_path = exec_dir + "/../data/sk30Jij.txt";
    std::ifstream fin(jij_path);

    if (!fin.is_open())
    {
        std::cerr << "Error: Failed to open Jij file.\n";
        return 1;
    }

    for (int i = 0; i < N * N; ++i)
    {
        if (!(fin >> J[i]))
        {
            std::cerr << "Error: Failed to read enough data from file.\n";
            return 1;
        }
    }

    fin.close();

    std::vector<int> best_S;
    unsigned int idx = 0;

    launch_energy_kernel(J.data(), N, &idx);
    float min_energy = compute_energy_cpu(J.data(), N, idx);

    for (size_t j = 0; j < N; ++j)
    {
        std::cout << (((idx >> j) & 1) ? 1 : -1) << ' ';

    }
    std::cout << std::endl;

    std::cout << min_energy << std::endl;

    return 0;
}
