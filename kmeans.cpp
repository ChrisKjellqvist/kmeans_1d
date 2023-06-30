//
// Created by Chris Kjellqvist on 6/30/23.
//

#include "kmeans.h"


#define my_max(a, b) ((a) > (b) ? (a) : (b))
#define my_min(a, b) ((a) < (b) ? (a) : (b))

// there will certainly be some bad corner cases if k ~ N, shouldn't occur otherwise, though
// In the larger algorithm, we're going to spoof removing the k-mean and then re-placing it in the same interval
std::pair<double, __fp16> get_mean_insert(int k, std::vector<__fp16> &means, const uint16_t *radix_bins, __fp16 min_data, __fp16 max_data) {
    // bottom and top are the minimum and maximum logical places we can place k
    //
    // if we are considering a mean that is between other means, then we only consider placing it anywhere between
    // its two neighboring means
    //
    // if it is an outlier k (top or bottom), then the logical limit is just the inner boundary +- data[inner], respectively
    if (K == 1) { //NOLINT
        double a = 0, b = 0, c = 0;
        for (int i = float2radix(min_data); i <= float2radix(max_data) + 1; ++i) {
            if (radix_bins[i] == 0) continue;
            a += radix_bins[i];
            b -= 2 * radix_bins[i] * (double)radix2float(i);
            c += radix_bins[i] * square((double)radix2float(i));
        }
        auto zero = -b / (2 * a);
        return {a * square(zero) + b * zero + c, __fp16(zero)};
    }
    __fp16 bottom, top;
    if (k == 0) {
        bottom = my_max(means[1] - 2 * (means[1] - min_data), 0);
    } else {
        bottom = means[k - 1];
    }

    if (k == K - 1) {
        top = my_min(means[K - 2] + 2 * (max_data - means[K - 2]), max_data);
    } else {
        top = means[k + 1];
    }

    auto midpoint = (top + bottom) / 2;
    uint32_t absolute_top_idx = float2radix(top);

    bool no_zero = true;
    // ax^2 + bx + c
    uint32_t a = 0;
    double b = 0, c = 0;

    uint32_t top_idx, bot_idx;
    { // limit the scope of i
        top_idx = float2radix(midpoint) + 1;
        bot_idx = float2radix(bottom);
        while (radix_bins[bot_idx] == 0) {
            bot_idx++;
        }
        while (radix_bins[top_idx] == 0 && top_idx <= absolute_top_idx) top_idx++;

        size_t i = bot_idx;
        for (; i < top_idx && radix2float(i) <= midpoint; ++i) {
            if (radix_bins[i] == 0) {
                continue;
            }
            // consider (x - datas[i])^2
            // x^2 - 2 * datas[i]x + datas[i] ^ 2
            a += radix_bins[i];
            b -= 2 * radix_bins[i] * (double)radix2float(i);
            c += square((double)radix2float(i)) * radix_bins[i];
        }
        for (; i < absolute_top_idx; ++i) {
            if (radix_bins[i] == 0) {
                continue;
            }
            double d = top - radix2float(i);
            c += square(d) * radix_bins[i];
        }
    }
    auto left_disc = bottom + 2 * (radix2float(bot_idx) - bottom);
    auto right_disc = top - 2 * (top - radix2float(top_idx));
    double left = bottom, right;
    // first discontinuity
    if (left_disc < right_disc) {
        right = left_disc;
    } else {
        right = right_disc;
    }
    // best_zero_.* are where we're going to put the k-mean for optimal placement
    double best_zero_position, best_zero_mag;
    double original_score;
    auto original_mean = means[k];
    int saw_zero = 0;

    // consider the discontinuities in order
    while (top_idx < absolute_top_idx || bot_idx < absolute_top_idx) {
        // d/dx of current parabola is 2ax + b
        // so solution = -b/2a
        auto derivative_zero = (-b) / (2 * a);
        if (left <= original_mean && original_mean <= right) {
            original_score = square((double)original_mean) * a + original_mean * b + c;
        }
        if (derivative_zero >= left && derivative_zero <= right) {
            saw_zero++;
            double par_at_dzero = derivative_zero * derivative_zero * a + derivative_zero * b + c;
            // x, y pair
            if (no_zero) {
                no_zero = false;
                best_zero_position = derivative_zero;
                best_zero_mag = par_at_dzero;
            } else if (par_at_dzero < best_zero_mag) {
                saw_zero++;
                best_zero_position = derivative_zero;
                best_zero_mag = par_at_dzero;
            }
        } // else this section does not have a local minimum
        left = right;
        if (left_disc < right_disc) {
            // move discontinuity bounds
            // if the disc comes the left then it's a parabola becoming flat and the ownership of this point is
            // transferred to the neighboring mean
            a -= radix_bins[bot_idx];
            b += 2 * (double)radix2float(bot_idx) * radix_bins[bot_idx];
            c -= radix_bins[bot_idx] * square((double)radix2float(bot_idx));
            auto diff = bottom - radix2float(bot_idx);
            c += square((double)diff) * radix_bins[bot_idx];

            do {
                bot_idx++;
            } while (radix_bins[bot_idx] == 0 && bot_idx < top_idx);
            if (bot_idx > top_idx) break;

            left_disc = bottom + 2 * (radix2float(bot_idx) - bottom);
        } else {
            // move discontinuity bounds
            // if the disc comes from the right then it's a flat area becoming a parabola
            a += radix_bins[top_idx];
            b -= 2 * (double)radix2float(top_idx) * radix_bins[top_idx];
            c += square((double)radix2float(top_idx)) * radix_bins[top_idx];
            auto diff = radix2float(absolute_top_idx) - radix2float(top_idx);
            c -= square((double)diff) * radix_bins[top_idx];
            do {
                top_idx++;
            } while (radix_bins[top_idx] == 0 && top_idx <= absolute_top_idx);
            right_disc = top - 2 * (top - radix2float(top_idx));
        }
        if (left_disc < right_disc) {
            right = left_disc;
        } else {
            right = right_disc;
        }
    }
    return {original_score - best_zero_mag, __fp16(best_zero_position)};
}
