// Chris Kjellqvist - 28 June 2023

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "util.h"
#include "kmeans.h"

double get_error(const std::vector<float_type> &means, const std::vector<float_type> &data) {
    double error = 0;
    for (auto &dat: data) {
        double min_dist = 1e9;
        for (auto &mean: means) {
            auto dist = abs((double)dat - (double)mean);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        error += min_dist;
    }
    return error / (double)data.size();
}


int main() {
    // make random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(18, 1);
    int num_not_converged = 0;
    int num_iterations_total = 0;
    double average_error = 0;

#ifndef HAS_FP16
    // sanity check the error propagation of the radix conversion
    for (int i = 0; i < 1000; ++i) {
        auto f = abs(dis(gen));
        auto r = float2radix(f);
        auto f2 = radix2float(r);
        if (abs(f - f2) > 0.1) {
            std::cout << "error in radix conversion: " << f << " " << r << " " << f2 <<  std::endl;
            throw std::runtime_error("error in radix conversion");
        }
    }

    std::cerr << "Passed sanity checks" << std::endl;
#endif

    std::vector<float_type> data;
    // generate data
    data.reserve(N_DATAS);
    for (int i = 0; i < N_DATAS; ++i) {
        data.push_back((float_type) dis(gen));
    }

    // total execution time counter
    unsigned long long total_time = 0;
    for (int R = 0; R < 768; ++R) {
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, 16, 3000);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // check if converged
        average_error += get_error(means, data);
    }

    std::cout << "average error: " << average_error / 768.0 << std::endl;
    std::cout << "total time: " << (total_time / (1000)) << "ms" << std::endl;
    std::cout << "time per iteration: " << total_time / 768 << "µs" << std::endl;
    std::cout << "Throughput: " << 768.0 / (total_time / 1000000.0) << " data points per second" << std::endl; //NOLINT
}
