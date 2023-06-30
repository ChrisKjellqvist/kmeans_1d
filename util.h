//
// Created by Chris Kjellqvist on 6/29/23.
//

#ifndef KMEANS_HACKING_UTIL_H
#define KMEANS_HACKING_UTIL_H

#include <cinttypes>
#include <cmath>

using radix_t = uint16_t;

#ifdef __ARM64_ARCH_8__
using halfFloat_t= __fp16;
#elif defined(__X86_64__) || defined(_M_X64)
using halfFloat_t = _Float16;
#else
#error "Unsupported platform"
#endif


#include "constants.h"

radix_t float2radix(halfFloat_t f);

halfFloat_t radix2float(radix_t radix);

halfFloat_t radix2float_exact(radix_t radix); // NOLINT

constexpr uint32_t n_radix_bins() {
    return 1 << (8 * sizeof(radix_t));
}


struct move_t {
    double improvement;
    halfFloat_t location;
    move_t(double imp, halfFloat_t loc): improvement(imp), location(loc) {}
    move_t() = default;
};

template <typename t>
t square(t q) { return q * q; }

template<typename t>
size_t binary_search(size_t lo, size_t hi, t *ar, t find) {
    auto guess = (lo + hi) >> 1;
    auto f = ar[guess];
    if (guess == lo || guess == hi) {
        if (f != find) return 0;
    }
    if (f == find) {
        // need to account (for this algorithm at least) for cases with duplicates. Need to find the lowest duplicate
        auto guess2 = guess - 1;
        while (true) {
            if (ar[guess2] == find) {
                guess = guess2;
                if (guess2 == 0) return guess;
                guess2--;
            } else return guess;
        }
    } else if (f < guess) {
        return binary_search(guess, hi, ar, find);
    } else return binary_search(lo, guess, ar, find);
}
#endif //KMEANS_HACKING_UTIL_H
