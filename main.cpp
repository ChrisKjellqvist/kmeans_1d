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
        // find closest mean and add error
        double smallest_error = 1e20;
        for (auto &mean: means) {
            auto dist = (double)dat - (double)mean;
            auto t_error = dist * dist;
            if (t_error < smallest_error) {
                smallest_error = t_error;
            }
        }
        error += smallest_error;
    }
    return error / (double)data.size();
}


int main() {
    // make random data
    std::random_device rd;
    std::mt19937 gen(rd());
    // generate random bools
    std::uniform_real_distribution<float> booler(0, 1);
    float center1 = 18, center2 = 45;
    float std_dev1 = 2, std_dev2 = 2;
    std::normal_distribution<float> dis1(center1, std_dev1);
    std::normal_distribution<float> dis2(center2, std_dev2);
    int num_not_converged = 0;
    int num_iterations_total = 0;
    double average_error = 0;

    // sanity check the error propagation of the radix conversion
    for (int i = 0; i < 1000; ++i) {
        auto f = dis1(gen) - MINIMUM_PERMISSIBLE_DATA_VALUE;
        auto r = float2radix(f);
        auto f2 = radix2float(r);
        if (abs(f - f2) > 0.1) {
            std::cout << "error in radix conversion: " << f << " " << r << " " << float(f2) <<  std::endl;
            throw std::runtime_error("error in radix conversion");
        }
    }

    std::cerr << "Passed sanity checks" << std::endl;

    std::vector<float_type> data;
    // generate data
    data.reserve(N_DATAS);
    for (int i = 0; i < N_DATAS; ++i) {
        if (booler(gen) < 0.5)
            data.push_back((float_type) dis1(gen));
        else
            data.push_back((float_type) dis2(gen));
    }

    // total execution time counter
    unsigned long long total_time = 0;
    for (int R = 0; R <
#ifndef NDEBUG
            1
#else
    768
#endif
    ; ++R) {
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, 16, 3000);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // check if converged
        average_error += get_error(means, data);
        // print means
#ifndef NDEBUG
        std::cout << "means: ";
        for (auto &mean: means) {
            std::cout << float(mean) << " ";
        }
        std::cout << std::endl;
        std::cout << "Inspect above means for centering around " << center1 << " and " << center2 << std::endl;
#endif
    }

    std::cout << "average error: " << average_error / 768.0 << std::endl;
    std::cout << "total time: " << (total_time / (1000)) << "ms" << std::endl;
    std::cout << "time per iteration: " << total_time / 768 << "µs" << std::endl;
    std::cout << "Throughput: " << 768.0 / (total_time / 1000000.0) << " data points per second" << std::endl; //NOLINT
}
