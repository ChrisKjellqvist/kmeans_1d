//
// Created by Chris Kjellqvist on 6/30/23.
//

#include "kmeans.h"
#include <iostream>
#include <cstring>

/**
 * Algorithm Description:
 *
 * Preprocessing:
 * DO NOT store the actual data points. Instead, store the distribution! This is useful because when using a highly
 * quantized datatype like fp16 and many data points, there are actually many collisions in the data set - ie many
 * data repeats! So we can store the distribution of the data points in a radix histogram, making the data representation
 * much more dense.
 * However, we still need to preserve the linearity of the radix relative to the datatype - this presents a problem.
 * Take for instance the positive FP16 numbers. They start at 0x0000 and go to 0x7FFF. You can actually use 2s-complement
 * math to increment to the next representable float - this works great as an index: simply use the floating point representation
 * as an index and you'll always get the next representable float. However, this doesn't work if you consider negative
 * floating point numbers because floating point representations have an explicit sign bit. There's really not much
 * of a way to work around this besides add the minimum data point to all of the data points, transforming all of them
 * into positive numbers. This essentially takes our 16-bit FP and turns it into a 15-bit with some precision loss for
 * the higher-order positive numbers in our range but this seems to be an acceptable sacrifice for the datasets that
 * are internally important to our group - they mostly exist as huge datasets (N ~ 50K) in the range -10 to 10.
 * This means that we end up with around 5K bins for 50K data points! This ends up saving us a lot down the line.
 *
 * Algorithm 1: K-mean twiddling
 * Description: Instead of performing a global placement of a mean, we consider the placement of a mean in a local
 * neighborhood. This neighborhood is defined by the two neighboring means. If the mean is an edge mean (ie the k=0,15
 * for K=16), then the neighborhood is defined by the minimum/maximum data point and the inner mean (k=1,14), respectively.
 * This makes intuitive sense because there is no value in placing a mean outside of the range of the data points.
 * Input: k, means, radix_bins, min_data, max_data
 * Output: (cost, mean) - minimize cost
 * 1. Compute the location of the neighborhood edges
 * 2. Start at the left edge of the neighborhood and begin to initialize our quadratic cost function
 * |  For each data point from left to right:
 * |  |  If the data point < midpoint between left and right edge, then the our cursor is IN the quadratic portion of
 * |  |     the cost function. Add the term (x - data_point)^2 to the quadratic cost function.
 * |  |  Else, the cursor is OUTSIDE the quadratic portion of the cost function. Add the term
 * |  |     (right_edge - data_point)^2 to the quadratic cost function.
 * 3. Initialize pointers bot_idx, top_idx to the left edges of data point sets for the left and right means.
 * For clarity, these should be the data point that is closest to the left mean and furthest from the right mean (but
 * still out of range of the left mean).
 * 4. Initialize the discontinuity range (left, right) to the location of the left mean and the first discontinuity.
 * Discontinuities in the cost function between (left_mean, right_mean) occur at left_mean + 2 * (data_point - left_mean)
 * for each data point left of the midpoint and at right_mean - 2 * (right_mean - data_point) for each data point right
 * of the midpoint. The right discontinuity is the first discontinuity that is greater than the left mean.
 * 5. Given the current state of the quadratic cost function and the discontinuity range, compute the zero of the
 * quadratic cost function's derivative to find the location of the minimum.
 * 6. If the zero is in the discontinuity range, record it if it is the minimum zero seen so far.
 * 7. Increment the discontinuity range from (left, right) to (right, next discontinuity). If the next discontinuity
 * comes from a data point that is now closer to the left mean than the cursor (we fall out of left_mean + 2 * diff),
 * then we subtract (left_mean - data_point)^2 from the quadratic cost function and add (data_point - left_mean)^2.
 * And conversely, if the next discontinuity comes from a data point that is now further from the right mean than it
 * is from the cursor, then we subtract (right_mean - data_point)^2 from the quadratic cost function and add
 * (data_point - x)^2.
 * 8. Repeat steps 5-7 until the discontinuity range is greater than the right mean.
 * 9. Return the minimum observed zero. There is guaranteed to be at least one minimum because adding a mean in between
 * two means will always decrease the cost function.
 *
 * Algorithm 2: Iterative K-mean placement
 * 1. Initialize the means to be uniformly distributed across the data range.
 * 2. For each mean:
 * |  Compute the improvement in the cost function if the mean is placed at the minimum of the cost function for its
 * |  neighborhood and store in an update table
 * 3. Select the greatest improvement in the update table and perform that k-mean update
 * 4. Mark the update as invalid in the table and update the table for indices k-1 and k+1 (if they exist) having
 * |  chosen to perform update k
 * 5. Repeat steps 2-4 until the update table contains no valid updates or potential improving placements
 * 6. Return the means
 */


// there will certainly be some bad corner cases if k ~ N, shouldn't occur otherwise, though
// In the larger algorithm, we're going to spoof removing the k-mean and then re-placing it in the same interval
std::pair<double, float_type> get_mean_insert(int k, size_t K, std::vector<float_type> &means, const uint16_t *radix_bins, float_type min_data, float_type max_data) {
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
        return {a * square(zero) + b * zero + c, float_type(zero)};
    }
    float_type bottom, top;
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
        if (a > 0) {
            // two tight clusters near the edges may result in a flat portion near the middle where placement of the
            // mean there will result in 0 points being assigned to the mean. This is _never_ optimal and will result
            // in a divide by zero error. At any rate, it's not important to consider besides for the obvious error.
            // it is also incredibly rare to have a flat portion in the middle, so this `if` will be trained unlikely
            auto derivative_zero = (-b) / (2 * a);
            if (left <= original_mean && original_mean <= right) {
                original_score = square((double) original_mean) * a + original_mean * b + c;
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
        }
        left = right;
        if (left_disc < right_disc) {
            // move discontinuity bounds
            // if the disc comes the left set then it's a parabola becoming flat and the ownership of this point is
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
            // if the disc comes from the right set then it's a flat area becoming a parabola
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
    return {original_score - best_zero_mag, float_type(best_zero_position)};
}


void preprocess_and_insert_data(const std::vector<float_type> &fpar, uint16_t *radix_bins, float_type &min_data, float_type &max_data) {
    min_data = fpar[0];
    max_data = fpar[0];
    for (const auto dat: fpar) {
        auto adjusted_datum = dat - MINIMUM_PERMISSIBLE_DATA_VALUE;
        if (dat < MINIMUM_PERMISSIBLE_DATA_VALUE) {
            throw std::runtime_error("data value (" + std::to_string(float(dat)) + ") smaller than minimum permissible data value (" + std::to_string(float(MINIMUM_PERMISSIBLE_DATA_VALUE)) + ")");
        }
        auto radix = float2radix(adjusted_datum);
        if (radix_bins[radix] == (1 << (sizeof(radix_t) * 8)) - 1) {
            std::cout << "radix bin overflow" << std::endl;
            throw std::runtime_error("radix bin overflow");
        }
        radix_bins[radix]++;
        if (dat < min_data) min_data = adjusted_datum;
        if (dat > max_data) max_data = adjusted_datum;
    }
}
#include <string>

// kmeans top-level function
std::vector<float_type> kmeans(const std::vector<float_type> &data, size_t K, size_t max_iterations) {
    uint16_t radix_bins[n_radix_bins()];

    float_type min_data, max_data;
    // location of 1-D means
    std::vector<float_type> means;
    means.reserve(K);

    // memset clear the bins
    memset(radix_bins, 0, sizeof(uint16_t) * n_radix_bins());

    preprocess_and_insert_data(data, radix_bins, min_data, max_data);

    // initialize means over the data points
    for (int i = 0; i < K; ++i) {
        means.push_back(static_cast<float_type>(i + 1) / (K + 1) * (max_data - min_data) + min_data);
    }
    // print initial means
#ifndef NDEBUG
    std::cout << "initial means: ";
    for (auto mean: means) {
        std::cout << float(mean) << " ";
    }
    std::cout << std::endl;
#endif

    std::pair<double, float_type> update_table[K];
    bool update_valid[K];
    for (int i = 0; i < K; ++i) {
        auto update = get_mean_insert(i, K, means, radix_bins, min_data, max_data);
        update_table[i].first = update.first;
        update_table[i].second = update.second;
        update_valid[i] = true;
    }

    int iterations = 0;
    // then refine
    while (true) {
        // find the best mean to move
        double min_improvement = 0;
        iterations++;
        int min_idx = 0;
        for (int j = 0; j < K; ++j) {
            if (update_table[j].first > min_improvement && update_valid[j]) {
                min_improvement = update_table[j].first;
                min_idx = j;
            }
        }
        if (min_improvement == 0 || means[min_idx] == update_table[min_idx].second) {
            break;
        }
        // move the mean
        means[min_idx] = update_table[min_idx].second;
        update_valid[min_idx] = false;
        // update the update table. First update neighbors, then update the moved mean
        if (min_idx > 0) {
            update_table[min_idx - 1] = get_mean_insert(min_idx - 1, K, means, radix_bins, min_data, max_data);
            update_valid[min_idx - 1] = update_table[min_idx - 1].second != means[min_idx - 1];
        }
        if (min_idx < K - 1) {
            update_table[min_idx + 1] = get_mean_insert(min_idx + 1, K, means, radix_bins, min_data, max_data);
            update_valid[min_idx + 1] = update_table[min_idx + 1].second != means[min_idx + 1];
        }
        if (iterations > max_iterations) {
            throw std::runtime_error("Failed to converge in " + std::to_string(max_iterations) + " iterations");
        }
    }
    // correct for offest in means
    for (auto &mean: means) {
        mean += MINIMUM_PERMISSIBLE_DATA_VALUE;
    }
    return means;
}