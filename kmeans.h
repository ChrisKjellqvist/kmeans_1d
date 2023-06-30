//
// Created by Chris Kjellqvist on 6/30/23.
//

#ifndef KMEANS_HACKING_KMEANS_H
#define KMEANS_HACKING_KMEANS_H

#include "util.h"
#include <vector>
#include <tuple>

std::pair<double, __fp16> get_mean_insert(int k, std::vector<__fp16> &means, const uint16_t *radix_bins, __fp16 min_data, __fp16 max_data);

#endif //KMEANS_HACKING_KMEANS_H
