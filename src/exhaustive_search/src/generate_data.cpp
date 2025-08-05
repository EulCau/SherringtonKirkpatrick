#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <cfloat>
#include <unistd.h>
#include <climits>
#include <string>
#include "solver.cuh"

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

void Jflat_to_Jmatrix(const std::vector<float>& J_flat, std::vector<float>& J, int N)
{
	size_t idx = 0;
	for (int i = 0; i < N; ++i)
	{
		for (int j = i + 1; j < N; ++j)
		{
			float val = J_flat[idx++];
			J[i * N + j] = val;
			J[j * N + i] = val;
		}
		J[i * N + i] = 0.0f;
	}
}

int main()
{
	constexpr int N = 30;
	constexpr size_t data_length = 10;
	constexpr float sigma = 0.2f;
	constexpr unsigned seed = 42;

	std::mt19937 rng(seed);
	std::normal_distribution<float> dist(0.0f, sigma);

	std::vector<float> J(N * N, 0.0f);
	std::vector<float> J_flat((N * (N - 1)) / 2);

	std::string output_path = "../data/sk_data.bin";
	std::ofstream fout(output_path, std::ios::binary);

	if (!fout.is_open())
	{
		std::cerr << "Error: Failed to open output file.\n";
		return 1;
	}

	std::cout << "Data generation start... " << std::endl;

	for (size_t sample = 0; sample < data_length; ++sample)
	{
		for (auto& val : J_flat) val = dist(rng);

		Jflat_to_Jmatrix(J_flat, J, N);

		unsigned int idx = 0;
		launch_energy_kernel(J.data(), N, &idx);
		float min_energy = compute_energy_cpu(J.data(), N, idx);

		fout.write(reinterpret_cast<const char*>(J_flat.data()), J_flat.size() * sizeof(float));
		fout.write(reinterpret_cast<const char*>(&min_energy), sizeof(float));

		if ((sample + 1) % 10 == 0 && (sample + 1) != data_length)
		{
			std::cout << "data " << sample + 1 << " generated;" << std::endl;
		}
		else if ((sample + 1) == data_length)
		{
			std::cout << "data " << sample + 1 << " generated." << std::endl;
		}
	}

	fout.close();
	std::cout << "Data generation complete: " << output_path << std::endl;

	return 0;
}
