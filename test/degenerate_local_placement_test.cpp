// Chris Kjellqvist - 28 June 2023

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "../src/util.h"
#include "../src/kmeans.h"
#include "../src/constants.h"

double get_error(const std::vector<double> &means, const std::vector<float_type> &data) {
    double error = 0;
    for (auto &dat: data) {
        // find closest mean and add error
        double smallest_error = 1e20;
        for (auto &mean: means) {
            auto dist = (double) dat - (double) mean;
            auto t_error = dist * dist;
            if (t_error < smallest_error) {
                smallest_error = t_error;
            }
        }
        error += smallest_error;
    }
    return error / (double) data.size();
}

/**
 * This test is designed to produce and then attempt to escape from a degenerate placement caused by a greedy
 * local placement.
 *
 * The problem is as follows (K = 2):
 * Consider a distribution with two primary peaks, one at -2, and one at 2. However, the magnitude of the peak at 2
 * is much smaller than the peak at -2. On the first iteration of the algorithm the means are set to be equally
 * spread over the data. If the spread of the data is very large compared to the spread of the peaks, then an initial
 * placement of the means may be such that the second mean is placed PAST the second peak. Then, when considering
 * where to place mean_0 between the leftmost datapoint and mean_1, the algorithm will place it at the second peak
 * because magnitude is overwhelming. Then, when attempting to place mean_1, we only consider between mean_0 and the
 * rightmost datapoint, ignoring the possibility of placing it at the first peak.
 *
 * We're going to try to escape this by using a global placement after finding a locally converged placement. If
 * this is not enough to converge to an optimal solution in all cases, then mean placement is non-convex and this
 * entire algorithm will not work. There is potential hope for doing a non-optimal placement and then hoping that its
 * converged loss will be less than the loss of the originally locally stable solution but again this would
 * imply that mean placement is non-convex in which case the best we'll be able to accomplish is "heuristic" k-means.
 */

int main() {
    // make random data
    std::random_device rd;
    std::mt19937 gen(rd());
    double average_error = 0;
    int N_DATAS = 50000;
    std::vector<float_type> data;
    std::vector<float> centers;
    std::vector<float> std_devs;
    std::vector<float> probs;

    centers.push_back(-4);
    std_devs.push_back(0.5);
    probs.push_back(0.4);

    centers.push_back(0);
    std_devs.push_back(0.5);
    probs.push_back(0.50);

    centers.push_back(4);
    std_devs.push_back(0.5);
    probs.push_back(0.1);

    std::vector<float> cumulative_probability;
    float acc = 0;
    for (auto &prob: probs) {
        acc += prob;
        cumulative_probability.push_back(acc);
    }

    {
        std::uniform_real_distribution<float> booler(0, 1);

        std::vector<std::normal_distribution<float> > distributions;
        for (int i = 0; i < centers.size(); ++i) {
            distributions.emplace_back(centers[i], std_devs[i]);
        }

        // generate data
        data.reserve(N_DATAS);
        for (int i = 0; i < N_DATAS; ++i) {
            auto r = booler(gen);
            int j = 0;
            for (;j < cumulative_probability.size() && r > cumulative_probability[j]; ++j) {}
            data.push_back(distributions[j](gen));
        }
    }

    uint16_t radix_bins[1 << 16];
    float_type data_min, data_max;
    preprocess_and_insert_data(data, radix_bins, data_min, data_max);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> means(2);
    means[0] = centers[1] - MINIMUM_PERMISSIBLE_DATA_VALUE;
    means[1] = centers[2] - MINIMUM_PERMISSIBLE_DATA_VALUE;
    double error_before =  get_error(means, data);

    std::cerr << "data_min: " << float(data_min) << std::endl;
    std::cerr << "data_max: " << float(data_max) << std::endl;

    std::cerr << "means: ";
    for (auto &mean: means) {
         std::cerr << (mean + MINIMUM_PERMISSIBLE_DATA_VALUE) << " ";
    }
    std::cerr << std::endl;

    find_global_placement(2, means, radix_bins, data_min, data_max);

    double error_after = get_error(means, data);

    std::cerr << "means after: ";
    for (auto &mean: means) {
         std::cerr << (mean + MINIMUM_PERMISSIBLE_DATA_VALUE) << " ";
    }
    std::cerr << std::endl;

    std::cout << "Error before: " << error_before << std::endl;
    std::cout << "Error after: " << error_after << std::endl;


}
