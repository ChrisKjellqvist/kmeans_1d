//
// Created by Chris Kjellqvist on 6/29/23.
//

#include "util.h"

// we want to organize floats by radix so that the index represented by the radix still represents a float ordering
radix_t float2radix(__fp16 f) {
    return reinterpret_cast<radix_t&>(f);
}

__fp16 radix2float_exact(radix_t radix) {
    __fp16 q = reinterpret_cast<__fp16&>(radix);
    return q + MINIMUM_DATA_VALUE;
}

__fp16 radix2float(radix_t f) {
    return reinterpret_cast<__fp16&>(f);
}

