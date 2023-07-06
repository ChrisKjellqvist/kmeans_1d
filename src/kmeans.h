//
// Created by Chris Kjellqvist on 6/30/23.
//

#ifndef KMEANS_HACKING_KMEANS_H
#define KMEANS_HACKING_KMEANS_H

#include "util.h"
#include <vector>
#include <tuple>

// there will certainly be some bad corner cases if k ~ N, shouldn't occur otherwise, though
// In the larger algorithm, we're going to spoof removing the k-mean and then re-placing it in the same interval
std::vector<double>
        kmeans(
                const std::vector<float_type> &data,
                int k,
                size_t max_iterations,
                bool *converged = nullptr);

bool find_global_placement(const int K,
                           std::vector<double> &means,
                           const uint16_t *radix_bins,
                           float_type min_data,
                           float_type max_data);

void preprocess_and_insert_data(const std::vector<float_type> &fpar, uint16_t *radix_bins, float_type &min_data,
                                float_type &max_data);

#endif //KMEANS_HACKING_KMEANS_H
