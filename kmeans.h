//
// Created by Chris Kjellqvist on 6/30/23.
//

#ifndef KMEANS_HACKING_KMEANS_H
#define KMEANS_HACKING_KMEANS_H

#include "util.h"
#include <vector>
#include <tuple>

std::pair<double, halfFloat_t> get_mean_insert(int k, std::vector<halfFloat_t> &means, const uint16_t *radix_bins, halfFloat_t min_data, halfFloat_t max_data);

#endif //KMEANS_HACKING_KMEANS_H
