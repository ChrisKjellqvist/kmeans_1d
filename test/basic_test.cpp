// Chris Kjellqvist - 28 June 2023

#include <iostream>
#include <vector>
#include <random>
#include "../src/util.h"
#include <chrono>
#include "../src/kmeans.h"
#include "../src/constants.h"

double get_error(const std::vector<double> &means, const std::vector<float_type> &data) {
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
    double average_error = 0;
    int N_DATAS = 50000;
    std::vector<float_type> data;
    float center1 = -2, center2 = 2;
    {
        std::uniform_real_distribution<float> booler(0, 1);
        float std_dev = 3;
        std::normal_distribution<float> dis1(center1, std_dev), dis2(center2, std_dev);
        // sanity check the error propagation of the radix conversion
        for (int i = 0; i < 1000; ++i) {
            auto f = dis1(gen) - MINIMUM_PERMISSIBLE_DATA_VALUE;
            auto r = float2radix((float_type)f);
            auto f2 = radix2float(r);
            if (abs(f - f2) > 0.1) {
                std::cout << "error in radix conversion: " << f << " " << r << " " << float(f2) << std::endl;
                throw std::runtime_error("error in radix conversion");
            }
        }
        std::cerr << "Passed conversion sanity checks" << std::endl;

        // generate data
        data.reserve(N_DATAS);
        for (int i = 0; i < N_DATAS; ++i) {
            if (booler(gen) < 0.5)
                data.push_back((float_type) dis1(gen));
            else
                data.push_back((float_type) dis2(gen));
        }
    }

    // find maximum and minimum datas
    float_type max_data = data[0];
    float_type min_data = data[0];
    for (auto &dat: data) {
        if (dat > max_data) max_data = dat;
        if (dat < min_data) min_data = dat;
    }
    std::cout << "SANITY: max data: " << float(max_data) << std::endl;
    std::cout << "SANITY: min data: " << float(min_data) << std::endl;

#ifndef NDEBUG
    int N_TIMES = 1;
#else
    int N_TIMES = 768;
#endif

    // total execution time counter
    unsigned long long total_time = 0;
    for (int R = 0; R < N_TIMES; ++R) {
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, 2, 3000);
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

    std::cout << "average error: " << average_error / N_TIMES << std::endl;
    std::cout << "time per kmean: " << total_time / N_TIMES << "µs" << std::endl;
    std::cout << "Throughput: " << N_TIMES / (total_time / 1000000.0) << " kmean per second" << std::endl; //NOLINT
}
