//
// Created by Chris Kjellqvist on 6/30/23.
//

#include "kmeans.h"
#include "constants.h"
#include <iostream>
#include <cstring>
#include <memory>
#include <algorithm> 

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
struct movement {
    double location;
    double improvement;
    bool valid;

    movement(double location, double improvement, bool valid) : location(location), improvement(improvement),
                                                                valid(valid) {}

    explicit movement() : location(0), improvement(0), valid(false) {}
};

movement
find_locally_optimal_placement(int k,
                               int K,
                               const std::vector<double> &means,
                               const uint16_t *radix_bins,
                               float_type min_data,
                               float_type max_data,
                               bool return_absolute_score) {
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
            b -= radix_bins[i] * (double) radix2float(i);
            c += radix_bins[i] * square((double) radix2float(i));
        }
        double zero = -b / (a);
        return movement(zero, a * square(zero) + 2 * b * zero + c, true);
    }
    double bottom, top;
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

    double midpoint = (top + bottom) / 2;
    uint32_t absolute_top_idx = float2radix((float_type) top);

    bool no_zero = true;
    // ax^2 + bx + c
    uint32_t a = 0;
    // b actually stores b / 2. Because we're only adding (x - c)^2, 2xc is the only thing being added to b
    // this ends up in wasting a mantissa bit which may end up being a problem and causing non-convergence
    double b = 0, c = 0;

    uint32_t top_idx, bot_idx;
    { // limit the scope of i
        top_idx = float2radix((float_type) midpoint) + 1;
        bot_idx = float2radix((float_type) bottom);
        while (radix_bins[bot_idx] == 0) bot_idx++;
        while (radix_bins[top_idx] == 0 && top_idx <= absolute_top_idx) top_idx++;

        size_t i = bot_idx;
        for (; i < top_idx && radix2float(i) <= midpoint; ++i) {
            if (radix_bins[i] == 0) {
                continue;
            }
            // consider (x - datas[i])^2
            // x^2 - 2 * datas[i]x + datas[i] ^ 2
            a += radix_bins[i];
            b -= radix_bins[i] * (double) radix2float(i);
            c += square((double) radix2float(i)) * radix_bins[i];
        }
        for (; i < absolute_top_idx; ++i) {
            if (radix_bins[i] == 0) {
                continue;
            }
            double d = top - radix2float(i);
            c += square(d) * radix_bins[i];
        }
    }
    double left_disc = bottom + 2 * (radix2float(bot_idx) - bottom);
    double right_disc = top - 2 * (top - radix2float(top_idx));
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
            auto derivative_zero = (-b) / (a);
            if (left <= original_mean && original_mean <= right && !return_absolute_score) {
                original_score = square((double) original_mean) * a + original_mean * 2 * b + c;
            }
            if (derivative_zero >= left - EPSILON && derivative_zero <= right + EPSILON) {
                saw_zero++;
                double par_at_dzero = derivative_zero * derivative_zero * a + derivative_zero * 2 * b + c;
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
            b += (double) radix2float(bot_idx) * radix_bins[bot_idx];
            c -= radix_bins[bot_idx] * square((double) radix2float(bot_idx));
            auto diff = bottom - radix2float(bot_idx);
            c += square((double) diff) * radix_bins[bot_idx];

            do {
                bot_idx++;
            } while (radix_bins[bot_idx] == 0 && bot_idx < top_idx);
            if (bot_idx > top_idx) break;

            left_disc = bottom + 2 * (radix2float(bot_idx) - bottom);
        } else {
            // move discontinuity bounds
            // if the disc comes from the right set then it's a flat area becoming a parabola
            a += radix_bins[top_idx];
            b -= (double) radix2float(top_idx) * radix_bins[top_idx];
            c += square((double) radix2float(top_idx)) * radix_bins[top_idx];
            auto diff = radix2float(absolute_top_idx) - radix2float(top_idx);
            c -= square((double) diff) * radix_bins[top_idx];
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
    if (saw_zero == 0) {
        // no data points in this section...
#ifndef NDEBUG
        std::cerr << "no data points in this section" << std::endl;
#endif
        return movement(0, original_mean, false);
    }
    // FIXME the following two checks are disabled for now
//    if (means[k] == best_zero_position) {
//        if (original_score != best_zero_mag) {
//            throw std::runtime_error("unexpected error 0");
//        }
//    }
    // this error should not occur when using native fp16 but when
    // using fp32, the error seems unavoidable due to error propagation
    // when computive radices
//    if (original_score < best_zero_mag) {
//        throw std::runtime_error("unexpected error 1");
//    }

    return movement(best_zero_position,
                    return_absolute_score ? best_zero_mag : original_score - best_zero_mag,
                    true);
}

bool find_global_placement(const int K,
                           std::vector<double> &means,
                           const uint16_t *radix_bins,
                           const float_type min_data,
                           const float_type max_data) {
    // test if the mean is better placed in a different cluster
    // We need this because we may have a situation where we reach convergence but due to the outer algorithm's
    // local placement, reaches a local minimum that isn't globally optimal. This is particularly common when
    // K is low

    // step 1: get the scores of each span between kmeans (and the endpoints)
    std::vector<double> section_scores(K + 1);
    auto start_idx = float2radix(min_data);
    auto end_idx = float2radix(max_data);
    double current_section_point = means[0];
    int current_section_idx = 0;
    // handle outlier spans
    for (auto start_idx = float2radix(min_data); start_idx < float2radix(means[0]); start_idx++) {
        section_scores[0] += square((double) radix2float(start_idx) - means[0]) * radix_bins[start_idx];
    }
    for (auto start_idx = float2radix(means[K - 1]); start_idx < float2radix(max_data); start_idx++) {
        section_scores[K] += square((double) radix2float(start_idx) - means[K - 1]) * radix_bins[start_idx];
    }
    // get inner scores
    for (int k = 0; k <= K - 1; ++k) {
        auto midpoint_idx = float2radix((means[k] + means[k + 1]) / 2);
        for (auto start_idx = float2radix(means[k]); start_idx < midpoint_idx; start_idx++) {
            section_scores[k+1] += square((double) radix2float(start_idx) - means[k]) * radix_bins[start_idx];
        }
        for (auto start_idx = midpoint_idx; start_idx < float2radix(means[k + 1]); start_idx++) {
            section_scores[k+1] += square((double) radix2float(start_idx) - means[k + 1]) * radix_bins[start_idx];
        }
    }

    std::vector<std::pair<int, double>> cost_to_remove_k;
    for (int k = 0; k < K; ++k) {
        // compute cost to remove k
        double cost = 0;
        if (k == 0) {
            for (int i = float2radix(min_data); i < float2radix(means[1]); ++i) {
                cost += square((double) radix2float(i) - means[1]) * radix_bins[i];
            }
        } else if (k == K - 1) {
            for (int i = float2radix(means[K - 2]); i < float2radix(max_data); ++i) {
                cost += square((double) radix2float(i) - means[K - 2]) * radix_bins[i];
            }
        } else {
            for (int i = float2radix(means[k - 1]); i < float2radix(means[k + 1]); ++i) {
                cost += square((double) radix2float(i) - means[k - 1]) * radix_bins[i];
            }
        }
        cost -= section_scores[k];
        cost -= section_scores[k + 1];
        cost_to_remove_k.emplace_back(k, cost);
    }
    // the most promising candidates for moving are those that are the least expensive to remove
    std::sort(cost_to_remove_k.begin(), cost_to_remove_k.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                  return a.second < b.second;
              });
    // while other means may be able to move, we only consider the best candidate because it is sufficient to do this
    // otherwise, a globally optimal solution would require O(K^2) time (ignoring factors of N)
    const auto candidate = cost_to_remove_k[0];
    // find the best placement for this candidate
    double best_improvement = 0;
    int best_mean_replacement = -1;
    double best_location = 0;
    for (int new_placement = 0; new_placement < K; ++new_placement) {
        if (new_placement == candidate.first) continue;
        std::vector<double> mean_with_dummy_k(K);
        for (int i = 0; i < K; ++i) {
            mean_with_dummy_k[i] = means[i];
        }
        mean_with_dummy_k[new_placement] = -1;
        mean_with_dummy_k[candidate.first] = means[new_placement];
        auto update = find_locally_optimal_placement(new_placement, K, mean_with_dummy_k,
                                                     radix_bins, min_data, max_data, true);
        double improvement = (section_scores[new_placement] - update.improvement) -
                             cost_to_remove_k[0].second;
        if (improvement > best_improvement) {
            best_improvement = improvement;
            best_mean_replacement = new_placement;
            best_location = update.location;
        }
    }
    if (best_mean_replacement != -1) {
        // we found a better placement
        means[candidate.first] = best_location;
        std::sort(means.begin(), means.end());
        return true;
    } else return false;
}

void preprocess_and_insert_data(const std::vector<float_type> &fpar, uint16_t *radix_bins, float_type &min_data,
                                float_type &max_data) {
    min_data = (float_type) fpar[0] - MINIMUM_PERMISSIBLE_DATA_VALUE;
    max_data = (float_type) fpar[0] - MINIMUM_PERMISSIBLE_DATA_VALUE;
    for (const auto dat: fpar) {
        auto adjusted_datum = float_type(dat - MINIMUM_PERMISSIBLE_DATA_VALUE);
        if (dat < MINIMUM_PERMISSIBLE_DATA_VALUE) {
            throw std::runtime_error(
                    "data value (" + std::to_string(float(dat)) + ") smaller than minimum permissible data value (" +
                    std::to_string(float(MINIMUM_PERMISSIBLE_DATA_VALUE)) + ")");
        }
        auto radix = float2radix(adjusted_datum);
        if (radix_bins[radix] == (1 << (sizeof(radix_t) * 8)) - 1) {
            std::cout << "radix bin overflow" << std::endl;
            throw std::runtime_error("radix bin overflow");
        }
        radix_bins[radix]++;
        if (adjusted_datum < min_data) min_data = adjusted_datum;
        if (adjusted_datum > max_data) max_data = adjusted_datum;
    }
}

#include <string>

// kmeans top-level function
std::vector<double>
kmeans(
        const std::vector<float_type> &data,
        int K,
        size_t max_iterations) {
    uint16_t radix_bins[n_radix_bins()];

    float_type min_data, max_data;
    // location of 1-D means
    std::vector<double> means;
    means.reserve(K);

    // memset clear the bins
    memset(radix_bins, 0, sizeof(uint16_t) * n_radix_bins());

    preprocess_and_insert_data(data, radix_bins, min_data, max_data);

    // initialize means over the data points
    for (int i = 0; i < K; ++i) {
        means.push_back(double(i + 1) / (K + 1) * (max_data - min_data) + min_data);
    }

    // print initial means
#ifndef NDEBUG
    std::cout << "initial means: ";
    for (auto mean: means) {
        std::cout << (float(mean) + MINIMUM_PERMISSIBLE_DATA_VALUE) << " ";
    }
    std::cout << std::endl;
#endif

    bool movement_candidate[K];
    for (int i = 0; i < K; ++i) {
        movement_candidate[i] = true;
    }

    bool have_globally_placed = false;
    int iterations = 0;
    while (true) {
        // find the best mean to move
        bool moved_something = false;
        for (int i = 0; i < K; ++i) {
            if (movement_candidate[i]) {
                auto movement_result = find_locally_optimal_placement(i, K, means, radix_bins, min_data, max_data,
                                                                      false);
                movement_candidate[i] = false;
                if (movement_result.valid && movement_result.improvement > 0) {
                    moved_something = true;
#ifndef NDEBUG
                    std::cerr << "moved mean " << i << " from " << (means[i] + MINIMUM_PERMISSIBLE_DATA_VALUE) << " to " << (movement_result.location + MINIMUM_PERMISSIBLE_DATA_VALUE)
                              << " with improvement " << movement_result.improvement << std::endl;
#endif
                    means[i] = movement_result.location;
                    if (i > 0) {
                        movement_candidate[i - 1] = true;
                    }
                    if (i < K - 1) {
                        movement_candidate[i + 1] = true;
                    }
                } else {
#ifndef NDEBUG
                    std::cerr << "mean " << i << " at " << (means[i] + MINIMUM_PERMISSIBLE_DATA_VALUE) << " is already optimal" << std::endl;
#endif
                }
            }
        }
        iterations++;


        if (!moved_something || iterations >= max_iterations) {
            // without: total loss: 0.0146769
            // with:    total loss: 0.0146985
            // if we have found a local minimum for local placement, test to see if we can find a better global placement
            // if we can't, we're done
            if (have_globally_placed) {
                break;
            }
            if (!find_global_placement(K, means, radix_bins, min_data, max_data)) {
                break;
            }
            iterations = 0;
            have_globally_placed = true;
        }
    }
    // correct for offest in means
    for (auto &mean: means) {
        mean += MINIMUM_PERMISSIBLE_DATA_VALUE;
    }
    return means;
}
