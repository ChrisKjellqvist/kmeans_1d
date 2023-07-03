// Chris Kjellqvist - 28 June 2023

#include <iostream>
#include <vector>
#include <random>
#include "../src/util.h"
#include <chrono>
#include "../src/kmeans.h"
#include "../src/constants.h"

double get_error(const std::vector<double> &means, const std::vector<float_type> &data) {
    double error = 0;
    for (auto &dat: data) {
        // find closest mean and add error
        double smallest_error = 1e20;
        for (auto &mean: means) {
            auto dist = (double)dat - (double)mean;
            auto t_error = dist * dist;
            if (t_error < smallest_error) {
                smallest_error = t_error;
            }
        }
        error += smallest_error;
    }
    return error / (double)data.size();
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

    centers.push_back(-2);
    probs.push_back(0.3);
    std_devs.push_back(3);

    centers.push_back(20);
    std_devs.push_back(0.5);
    probs.push_back(0.65);

    centers.push_back(40);
    std_devs.push_back(0.5);
    probs.push_back(0.5);

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
            if (booler(gen) < 0.5)
                data.push_back((float_type) dis1(gen));
            else
                data.push_back((float_type) dis2(gen));
        }
    }

    // find maximum and minimum datas
    float_type max_data = data[0];
    float_type min_data = data[0];
    for (auto &dat: data) {
        if (dat > max_data) max_data = dat;
        if (dat < min_data) min_data = dat;
    }
    std::cout << "SANITY: max data: " << float(max_data) << std::endl;
    std::cout << "SANITY: min data: " << float(min_data) << std::endl;

#ifndef NDEBUG
    int N_TIMES = 1;
#else
    int N_TIMES = 768;
#endif

    // total execution time counter
    unsigned long long total_time = 0;
    for (int R = 0; R < N_TIMES; ++R) {
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, 2, 3000);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // check if converged
        average_error += get_error(means, data);
        // print means
#ifndef NDEBUG
        std::cout << "means: ";
        for (auto &mean: means) {
            std::cout << float(mean) << " ";
        }
        std::cout << std::endl;
        std::cout << "Inspect above means for centering around " << center1 << " and " << center2 << std::endl;
#endif
    }

    std::cout << "average error: " << average_error / N_TIMES << std::endl;
    std::cout << "time per kmean: " << total_time / N_TIMES << "µs" << std::endl;
    std::cout << "Throughput: " << N_TIMES / (total_time / 1000000.0) << " kmean per second" << std::endl; //NOLINT
}
