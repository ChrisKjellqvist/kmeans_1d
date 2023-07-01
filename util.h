//
// Created by Chris Kjellqvist on 6/29/23.
//

#ifndef KMEANS_HACKING_UTIL_H
#define KMEANS_HACKING_UTIL_H

#include <cinttypes>
#include <cmath>

#define my_max(a, b) ((a) > (b) ? (a) : (b))
#define my_min(a, b) ((a) < (b) ? (a) : (b))


using radix_t = uint16_t;

#include "constants.h"

radix_t float2radix(float_type f);

float_type radix2float(radix_t radix);

constexpr uint32_t n_radix_bins() {
    return 1 << (8 * sizeof(radix_t));
}

template <typename t>
t square(t q) { return q * q; }

#endif //KMEANS_HACKING_UTIL_H
