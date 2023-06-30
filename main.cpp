// Chris Kjellqvist - 28 June 2023

#include <iostream>
#include <vector>
#include "util.h"
#include "kmeans.h"


void preprocess_and_insert_data(std::vector<__fp16> &fpar, uint16_t *radix_bins, __fp16 &min_data, __fp16 &max_data) {
    min_data = fpar[0];
    max_data = fpar[0];
    for (auto &dat: fpar) {
        auto radix = float2radix(dat - MINIMUM_DATA_VALUE);
        radix_bins[radix]++;
        if (dat < min_data) min_data = dat;
        if (dat > max_data) max_data = dat;
    }
}

double get_error(const std::vector<__fp16> &means, const std::vector<__fp16> &data) {
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
    return error / data.size();
}

#include <random>

int main() {
    // make random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(18, 1);
    int num_not_converged = 0;
    int num_iterations_total = 0;

    // total execution time counter
    unsigned long long total_time = 0;
    for (int R = 0; R < 768; ++R) {
        uint16_t radix_bins[n_radix_bins()];

        std::vector<__fp16> data;
        __fp16 min_data, max_data;
        // location of 1-D means
        std::vector<__fp16> means;
        for (auto &radix_bin: radix_bins) {
            radix_bin = 0;
        }

        // generate data
        data.reserve(N_DATAS);
        for (int i = 0; i < N_DATAS; ++i) {
            data.push_back((__fp16) dis(gen));
        }
        preprocess_and_insert_data(data, radix_bins, min_data, max_data);

        // initialize means over the data points
        means.reserve(K);
        for (int i = 0; i < K; ++i) {
            means.push_back(static_cast<__fp16>(i + 1) / (K + 1) * (max_data - min_data) + min_data);
        }


        auto t1 = std::chrono::high_resolution_clock::now();

        std::pair<double, __fp16> update_table[K];
        bool update_valid[K];
        for (int i = 0; i < K; ++i) {
            update_table[i] = get_mean_insert(i, means, radix_bins, min_data, max_data);
            update_valid[i] = true;
        }

        int iterations = 0;
        // then refine
        while (true) {
            // find the best mean to move
            double min_improvement = 0;
            iterations++;
            int min_idx = 0;
            for (int j = 0; j < K; ++j) {
                if (update_table[j].first > min_improvement && update_valid[j]) {
                    min_improvement = update_table[j].first;
                    min_idx = j;
                }
            }
            if (min_improvement == 0 || means[min_idx] == update_table[min_idx].second) {
                num_iterations_total += iterations;
                break;
            }
            // move the mean
            means[min_idx] = update_table[min_idx].second;
            update_valid[min_idx] = false;
            // update the update table. First update neighbors, then update the moved mean
            if (min_idx > 0) {
                update_table[min_idx - 1] = get_mean_insert(min_idx - 1, means, radix_bins, min_data, max_data);
                update_valid[min_idx - 1] = update_table[min_idx - 1].second != means[min_idx - 1];
            }
            if (min_idx < K - 1) {
                update_table[min_idx + 1] = get_mean_insert(min_idx + 1, means, radix_bins, min_data, max_data);
                update_valid[min_idx + 1] = update_table[min_idx+ 1].second != means[min_idx + 1];
            }
            if (iterations > 1000) {
                num_not_converged++;
                num_iterations_total += iterations;
                // print out current update table and break
//                std::cout << "iterations: " << iterations << std::endl;
//                for (int i = 0; i < K; ++i) {
//                    std::cout << float(update_table[i].location) << " improvement: " << float(update_table[i].improvement) << " " << update_valid[i] << "\t";
//                }
//                std::cout << std::endl;
                break;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        // print final error
//        auto error = get_error(means, data);
//        std::cerr << "error: " << error << std::endl;

        // print out final table
//        std::cout << "iterations: " << iterations << std::endl;
    }

    std::cout << "num not converged: " << num_not_converged << std::endl;
    std::cout << "average iterations: " << num_iterations_total / 768.0 << std::endl;
    std::cout << "total time: " << (total_time / (1000)) << "ms" << std::endl;
    // time per iteration
    std::cout << "time per iteration: " << total_time / 768 << "µs" << std::endl;
    std::cout << "Throughput: " << 768.0 / (total_time / 1000000.0) << " data points per second" << std::endl; //NOLINT
}