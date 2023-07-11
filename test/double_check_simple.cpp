#include <iostream>
#include "../src/constants.h"
#include "../src/kmeans.h"
#include "ckmeans_wrapper.h"
#include <gmp.h>

double compute_one_loss(float_type data_point, const std::vector<double> &centroids) {
    double smallest_error = 1e20;
    for (auto &mean: centroids) {
        auto dist = (double) data_point - (double) mean;
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
    return error / (double) data.size();
}

#include <random>

int main() {
    int N_DATAS = 100000;
#define DSIZE 1000
    int K_MEANS = 3;

    for (int r = 0; r < N_DATAS; ++r) {
        std::random_device rd;
        auto seed = rd();
        std::mt19937 gen(173074745);
        std::uniform_real_distribution<float> dis1(0, 20);
        float center = 14, stddev = 2;
        std::normal_distribution<float> dis_norm(center, stddev);
        float prob = 0.5;
        std::bernoulli_distribution dis_bernoulli(prob);

        // sample_0 with shape [49869]
        auto data = std::vector<float>(DSIZE);
        for (int j = 0; j < DSIZE; ++j) {
            if (dis_bernoulli(gen)) {
                data[j] = dis_norm(gen); //NOLINT
            } else {
                data[j] = dis1(gen); //NOLINT
            }
        }
        auto *radixes = new uint16_t[1 << 16];
        memset(radixes, 0, sizeof(uint16_t) * (1 << 16));
        float_type data_min, data_max;
        std::vector<float> data_float;
        std::vector<float_type> data_truncated;
        for (auto &d: data) {
            auto truncated = radix2float(float2radix(d));
//            printf("d: %0.8f, truncated: %0.8f\n", d, truncated);
            data_truncated.push_back(truncated);
            data_float.push_back(truncated);
        }

        std::sort(data_float.begin(), data_float.end());
#ifndef NDEBUG
        printf("DATA: \n");
        for (auto &d: data_float) {
            printf("%0.8f\n", d);
        }
#endif

        // truncate precision to fp16
        auto ckmeans_means = ckmeans_wrapper(data_float, K_MEANS);
#ifndef NDEBUG
        printf("CKMEANS: \n");
        for (auto &d: ckmeans_means) {
            printf("%0.8f\n", d);
        }
#endif
        // confirm that global converged is maintained by CKKMeans
        preprocess_and_insert_data(data_truncated, radixes, data_min, data_max);
        mpf_set_default_prec(256);
        mpf_t original_score, dist, this_dist;
        mpf_inits(original_score, dist, this_dist, nullptr);

        for (int j = 0; j < K_MEANS; ++j) {
            auto r = find_locally_optimal_placement(j, K_MEANS, ckmeans_means, radixes, data_min, data_max, false);
            // find expected "original score"
            mpf_set_d(original_score, 0);
            for (auto &d: data_float) {
                int corresponding_centroid = -1;
                mpf_set_d(dist, 1e10);
                for (int k = 0; k < K_MEANS; ++k) {
                    mpf_set_d(this_dist, (double) d - (double) ckmeans_means[k]);
                    mpf_mul(this_dist, this_dist, this_dist);
                    if (mpf_cmp(this_dist, dist) < 0) {
                        mpf_set(dist, this_dist);
                        corresponding_centroid = k;
                    }
                }
                if (corresponding_centroid == j) {
                    mpf_add(original_score, original_score, dist);
                } else if (corresponding_centroid == j - 1 && d > ckmeans_means[j - 1]) {
                    mpf_add(original_score, original_score, dist);
                } else if (corresponding_centroid == j + 1 && d < ckmeans_means[j + 1]) {
                    mpf_add(original_score, original_score, dist);
                }
            }
#ifndef NDEBUG
            printf("expected original score: %0.16f\n", mpf_get_d(original_score));
#endif
            if (r.valid) {
#ifndef NDEBUG
                std::cout << r.improvement << " " << ckmeans_means[j] << " -> " << r.location << std::endl;
#endif
                if (r.improvement > 1e-8) {
                    printf("seed is %lu\n", seed);
                    fflush(stdout);
                    throw std::runtime_error("CKMeans did not converge");
                }
#ifndef NDEBUG
                std::cout << "-------------------------------" << std::endl;
#endif
            }
#ifndef NDEBUG
            std::cout << "=========================================================" << std::endl;
#endif
        }
    }
    std::cout << "Seems that CKMeans converged for all " << N_DATAS << " data sets." << std::endl;

    return 0;
}
