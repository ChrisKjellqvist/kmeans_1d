//
// Created by Chris Kjellqvist on 6/29/23.
//

#ifndef KMEANS_HACKING_UTIL_H
#define KMEANS_HACKING_UTIL_H

#include <cinttypes>
#include <cmath>

#define my_max(a, b) ((a) > (b) ? (a) : (b))
#define my_min(a, b) ((a) < (b) ? (a) : (b))
#define my_fabs(a) ((a) < 0 ? -(a) : (a))

using radix_t = uint16_t;

//#ifdef __ARM64_ARCH_8__
//#define HAS_FP16
//using float_type = __fp16;
//#else
using float_type = float;
//#endif

radix_t float2radix(float_type f);

double radix2float(radix_t radix);

constexpr uint32_t n_radix_bins() {
    return 1 << (8 * sizeof(radix_t));
}

template <typename t>
t square(t q) { return q * q; }

#endif //KMEANS_HACKING_UTIL_H
