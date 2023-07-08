#include <iostream>
#include "../src/constants.h"
#include "../src/kmeans.h"
#include "ckmeans_wrapper.h"

double compute_one_loss(float_type data_point, const std::vector<double> &centroids) {
    double smallest_error = 1e20;
    for (auto &mean: centroids) {
        auto dist = (double)data_point - (double)mean;
        dist = fabs(dist);
        if (dist < smallest_error) {
            smallest_error = dist;
        }
    }
    return smallest_error;
}

double compute_loss(const std::vector<float_type> &data, const std::vector<double> &centroids) {
    double error = 0;
    for (auto &dat: data) {
        // find closest mean and add error
        error += compute_one_loss(dat, centroids);
    }
    // Average error
    return error / (double)data.size();
}
#include <random>

int main() {
    std::random_device rd;
    auto seed = rd();
    std::cout << "seed is " << seed << std::endl;
    std::mt19937 gen(181224946);
    std::uniform_real_distribution<float> dis1(0, 20);
    float center = 14, stddev = 2;
    std::normal_distribution<float> dis_norm(center, stddev);
    float prob = 0.5;
    std::bernoulli_distribution dis_bernoulli(prob);
#define DSIZE 6

#ifndef NDEBUG
    int N_DATAS = 1;
#else
    int N_DATAS = 150;
#endif

    int K_MEANS = 3;

    double largest_scaled_error = 0;
    double largest_error_val = 0;
    for (int i = 0; i < 1000; ++i) {
        auto f = dis1(gen) - MINIMUM_PERMISSIBLE_DATA_VALUE;
        auto r = float2radix((float_type)f);
        auto f2 = radix2float(r);
        // extract exponent
#ifdef HAS_FP16
        uint16_t exponent = (reinterpret_cast<uint16_t &>(r) >> 10) & 0x1F;
        auto scaled_error = fabs(f - f2) / (1 << exponent);
        if (scaled_error > largest_scaled_error) {
            largest_scaled_error = scaled_error;
            largest_error_val = f;
        }
#else
        std::cerr << "sanity not implemented yet for platforms without fp16" << std::endl;
        break;
#endif
    }
    if (largest_scaled_error > 1.0 / (1 << 10)) {
        std::cout << "Largest error is " << largest_scaled_error << std::endl;
        std::cout << "Largest expected error is " << 1.0 / (1 << 10) << std::endl;
        std::cout << "Largest error val is " << largest_error_val << std::endl;
        throw std::runtime_error("Largest error is too large");
    }
    std::cerr << "Passed conversion sanity checks" << std::endl;


    for (int i = 0; i < N_DATAS; i++) {
        // sample_0 with shape [49869]
        auto data = std::vector<float_type>(DSIZE);
        for (int j = 0; j < DSIZE; ++j) {
            if (dis_bernoulli(gen)) {
                data[j] = dis_norm(gen); //NOLINT
            } else {
                data[j] = dis1(gen); //NOLINT
            }
        }
        // truncate precision to fp16
        limit_precision_to_fp16(data.data(), data.size()); //NOLINT
        std::vector<float> data_float(data.begin(), data.end());
        auto ckmeans_means = ckmeans_wrapper(data_float, K_MEANS);

        auto *radixes = new uint16_t[1 << 16];
        memset(radixes, 0, sizeof(uint16_t) * (1 << 16));
        float_type data_min, data_max;
        preprocess_and_insert_data(data, radixes, data_min, data_max);
        // add global offset to CKMeans results
        for(auto &r: ckmeans_means) r -= MINIMUM_PERMISSIBLE_DATA_VALUE;

        // confirm that global converged is maintained by CKKMeans
        for (int j = 0; j < K_MEANS; ++j) {
            auto r = find_locally_optimal_placement(j, K_MEANS, ckmeans_means, radixes, data_min, data_max, false);
            // find expected "original score"
            double original_score = 0;
            for (auto &d: data) {
                auto dist = 1e20;
                int corresponding_centroid = -1;
                dist = fabs(dist);
                for (int k = 0; k < K_MEANS; ++k) {
                    auto actual_mean = ckmeans_means[k] + MINIMUM_PERMISSIBLE_DATA_VALUE;
                    auto this_dist = (double)d - (double)actual_mean;
                    this_dist = this_dist * this_dist;
                    if (this_dist < dist) {
                        dist = this_dist;
                        corresponding_centroid = k;
                    }
                }
                if (corresponding_centroid == j) {
                    original_score += dist;
                } else if (corresponding_centroid == j - 1 && d > ckmeans_means[j - 1] + MINIMUM_PERMISSIBLE_DATA_VALUE) {
                    original_score += dist;
                } else if (corresponding_centroid == j + 1 && d < ckmeans_means[j + 1] + MINIMUM_PERMISSIBLE_DATA_VALUE) {
                    original_score += dist;
                }
            }
//            std::cout << "expected original score: " << original_score << std::endl;
            if (r.valid) {
                std::cout << r.improvement << " " << ckmeans_means[j] << " -> " << r.location << std::endl;
                if (r.improvement > 1e-6) {
                    throw std::runtime_error("CKMeans did not converge");
                }
            }
        }
    }
    std::cout << "Seems that CKMeans converged for all " << N_DATAS << " data sets." << std::endl;

    return 0;
}
