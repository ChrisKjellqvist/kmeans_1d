//
// Created by Chris Kjellqvist on 6/30/23.
//

#ifndef KMEANS_HACKING_KMEANS_H
#define KMEANS_HACKING_KMEANS_H

#include "util.h"
#include <vector>
#include <tuple>

std::vector<float_type> kmeans(std::vector<float_type> &data, size_t k, size_t max_iterations);

#endif //KMEANS_HACKING_KMEANS_H
