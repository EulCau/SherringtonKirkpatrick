#include <iostream>
#include <vector>
#include "solver.cuh"

int main() {
    constexpr int N = 30;
    std::vector<float> J(N * N, 0.0f);

    // 填入自己的 J 数据（这里用随机数举例）
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            J[i * N + j] = (i + j) % 5 - 2;

    std::vector<int> best_S;
    float best_E;

    solve_exhaustive_gpu(J.data(), N, best_S, best_E);

    std::cout << "Best energy: " << best_E << "\n";
    std::cout << "Best configuration:\n";
    for (int i = 0; i < N; ++i)
        std::cout << best_S[i] << " ";
    std::cout << "\n";

    return 0;
}
