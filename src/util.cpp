//
// Created by Chris Kjellqvist on 6/29/23.
//

#include "util.h"
#include <cstdio>

// we want to organize floats by radix so that the index represented by the radix still represents a float ordering
radix_t float2radix(float_type f) {
#ifdef HAS_FP16
    radix_t idx = reinterpret_cast<radix_t&>(f);
    bool sign = (idx & 0x8000) >> 15;
    if (sign) {
        idx = ~idx;
    } else {
        idx = idx | 0x8000;
    }
    return idx;
#else
    uint32_t f32 = reinterpret_cast<uint32_t&>(f);
    auto exp = int32_t((f32 >> 23) & 0xFF);
    uint32_t mantissa = (f32 & 0x7FFFFF) >> (23 - 10);
    // do we round up?
    bool round_up = f32 & 0x1FFF;
    // exp is 8 bits, need to transform it to 5 bits. Can't directly shift it unfortunately
    // We need to do some actual arithmetic. Convert to signed exponent and then back to unsigned
    // except then for fp16
    exp = my_max(exp - 127 + 15, 0);
    return ((exp << 10) |  mantissa) + round_up;
#endif
}
#include <cstdio>

float_type radix2float(radix_t f) {
#ifdef HAS_FP16
    uint16_t tmp;
    bool sign = (f & 0x8000) >> 15;
    if (sign) {
        tmp = f & 0x7FFF;
    } else {
        tmp = ~f;
    }
    return reinterpret_cast<float_type&>(tmp);
#else
    uint32_t exp = (f >> 10) & 0x1F;
    uint32_t mantissa = f & 0x3FF;
    // exp is 5 bits, need to transform it to 8 bits
    exp = exp - 15 + 127;
    uint32_t f32 = (exp << 23) | (mantissa << (23 - 10));
    return reinterpret_cast<float_type&>(f32);
#endif
}

void limit_precision_to_fp16(double *data, int N) {
    for (int i = 0; i < N; i++) {
        data[i] = radix2float(float2radix(data[i]));
    }
}

void limit_precision_to_fp16(float *data, int N) {
    for (int i = 0; i < N; i++) {
        data[i] = radix2float(float2radix(data[i]));
    }
}
#ifdef HAS_FP16
void limit_precision_to_fp16(__fp16 *data, int N) {}
#endif
