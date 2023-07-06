//
// Created by Chris Kjellqvist on 7/6/23.
//

#include "ckmeans_wrapper.h"
#include "Ckmeans.1d.dp.h"

std::vector<double> ckmeans_wrapper(const std::vector <float> &data, int K) {
    auto cluster = new int[data.size()];
    auto means = new double[K];
    auto withinss = new double[K];
    auto size = new double[K];
    auto BICs = new double[K];
    std::string estimate_k = "false";
    std::string method = "loglinear";
    auto data_doubles = new double[data.size()];
    for (int i = 0; i < data.size(); i++) {
        data_doubles[i] = (double)data[i];
    }
    kmeans_1d_dp(data_doubles, data.size(), nullptr, K, K, cluster, means, withinss, size, BICs, estimate_k, method, DISSIMILARITY::L2);
    std::vector<double> real_means(K);
    for (int i = 0; i < K; i++) {
        real_means[i] = means[i];
    }
    delete[] cluster;
    delete[] means;
    delete[] withinss;
    delete[] size;
    delete[] BICs;

    return real_means;
}
