//
// Created by Chris Kjellqvist on 7/8/23.
//

#include "util.h"
#include <stdexcept>
#include <iostream>

int main() {
    radix_t before, after;
    before = 0;
    after = 1;
    // check that radix2float is monotonic
    for (; after < 0xFFFF; ++after, ++before) {
        if (radix2float(before) > radix2float(after)) {
            std::cout << "before: " << before << " after: " << after << std::endl;
            std::cout << "before: " << (float)radix2float(before) << " after: " << (float)radix2float(after) << std::endl;
            throw std::runtime_error("radix2float is not monotonic");
        }
    }

    // check that radix2float o float2radix is the identity
    for (radix_t i = 0; i < 0xFFFF; ++i) {
        if (float2radix(radix2float(i)) != i) {
            std::cout << "i: " << i << " " << std::hex << i << std::endl;
            float f = radix2float(i);
            std::cout << "r2f(i): " << f << " " << std::hex << reinterpret_cast<radix_t&>(f) << std::endl;
            std::cout << "i: " << float2radix(radix2float(i)) << std::endl;
            throw std::runtime_error("radix2float o float2radix is not the identity");
        }
    }
}