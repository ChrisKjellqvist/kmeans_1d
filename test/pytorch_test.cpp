#include <iostream>
#include <torch/torch.h>
#include "../src/constants.h"
#include "../src/kmeans.h"
#include "ckmeans_wrapper.h"

std::vector<char> get_the_bytes(const std::string &filename) {
    std::cout << "filename is " << filename << std::endl;
    std::ifstream input(filename, std::ios::binary);

    // Assert file exists
    std::stringstream ss;
    // assert can be ignored in release mode
    if (!input.is_open()) throw std::runtime_error(ss.str().c_str());

    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

std::vector<float_type> tensor_to_vector(const torch::Tensor &tensor) {
    std::vector<float_type> vec;
    vec.reserve(tensor.size(0));

    auto tensor_ac = tensor.accessor<float, 1>();
    for (int i = 0; i < tensor.size(0); i++) {
        auto this_item = tensor_ac[i];
        // cast to float_type
        auto casted = (float_type) this_item;
        vec.push_back(casted);
    }

    return vec;
}

double compute_one_loss(float_type data_point, const std::vector<double> &centroids) {
    double smallest_error = 1e20;
    for (auto &mean: centroids) {
        auto dist = (double)data_point - (double)mean;
        dist = dist * dist;
        if (dist < smallest_error) {
            smallest_error = dist;
        }
    }
    return smallest_error;
}

double compute_loss(const std::vector<float_type> &data, const std::vector<double> &centroids) {
    double error = 0;
    for (auto &dat: data) {
        // find closest mean and add error
        error += compute_one_loss(dat, centroids);
    }
    // Average error
    return error / (double)data.size();
}

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

int main() {
    std::vector<char> f = get_the_bytes("sample.pt");
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor sample = x.toTensor();

    long total_time = 0;
    double total_loss = 0;
    long total_ckmeans_time = 0;
    double total_ckmeans_loss = 0;
#ifndef NDEBUG
    int N_DATAS = 1;
#else
    int N_DATAS = 50;
#endif

    int K_MEANS = 16;

    for (int i = 0; i < N_DATAS; i++) {
        const auto sample_0 = sample.index(
                {torch::indexing::TensorIndex(0), torch::indexing::TensorIndex(torch::indexing::Slice()),
                 torch::indexing::TensorIndex(i)});
        // sample_0 with shape [49869]
        auto data = tensor_to_vector(sample_0);
        // truncate precision to fp16
        limit_precision_to_fp16(data.data(), data.size());
        bool converged;
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, K_MEANS, 3000, &converged);
        auto end = std::chrono::high_resolution_clock::now();
        if (!converged) {
            std::cout << "not converged" << std::endl;
        }
        auto loss = compute_loss(data, means);
        total_loss += loss;
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::vector<float> data_float(data.begin(), data.end());
        auto start_ckmeans = std::chrono::high_resolution_clock::now();
        auto ckmeans_means = ckmeans_wrapper(data_float, K_MEANS);
        auto end_ckmeans = std::chrono::high_resolution_clock::now();
        auto ckmeans_loss = compute_loss(data, ckmeans_means);
        total_ckmeans_loss += ckmeans_loss;
        total_ckmeans_time += std::chrono::duration_cast<std::chrono::microseconds>(end_ckmeans - start_ckmeans).count();

        // print means of both algortithms
//        std::cout << "means ckk:     ";
//        for (auto &mean: means) {
//            printf("%0.6f ", mean);
//        }
//        auto color = (loss < ckmeans_loss) ? GREEN : RED;
//        std::cout << color << "loss: " << loss << RESET;
//        std::cout << std::endl;
//        std::cout << "means ckmeans: ";
//        for (auto &mean: ckmeans_means) {
//            printf("%0.6f ", mean);
//        }
//        std::cout << YELLOW << "loss: " << ckmeans_loss << RESET;
//        std::cout << std::endl;
    }

    // print total time in second
    std::cout << "total time: " << (total_time / (1000)) << "ms" << std::endl;
    std::cout << "total loss: " << total_loss / 768.0 << std::endl;
    std::cout << "total ckmeans time: " << (total_ckmeans_time / (1000)) << "ms" << std::endl;
    std::cout << "total ckmeans loss: " << total_ckmeans_loss / 768.0 << std::endl;

    return 0;
}
