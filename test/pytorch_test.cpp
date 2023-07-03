#include <iostream>
#include <torch/torch.h>
#include "../src/constants.h"
#include "../src/kmeans.h"

std::vector<char> get_the_bytes(std::string &filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

std::vector<float_type> tensor_to_vector(torch::Tensor &tensor) {
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

double compute_one_loss(float_type data_point, std::vector<double> &centroids) {
    double smallest_error = 1e20;
    for (auto &mean: centroids) {
        auto dist = (double)data_point - (double)mean;
        auto t_error = dist > 0 ? dist : -dist;
        if (t_error < smallest_error) {
            smallest_error = t_error;
        }
    }
    return smallest_error;
}

double compute_loss(std::vector<float_type> &data, std::vector<double> &centroids) {
    double error = 0;
    for (auto &dat: data) {
        // find closest mean and add error
        error += compute_one_loss(dat, centroids);
    }
    // Average error
    return error / (double)data.size();
}

int main() {
    std::vector<char> f = get_the_bytes((std::string &) "sample.pt");
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor sample = x.toTensor();

    long total_time = 0;
    double total_loss = 0;
#ifndef NDEBUG
    int N_DATAS = 5;
#else
    int N_DATAS = 768;
#endif

    for (int i = 0; i < N_DATAS; i++) {
        std::cout << "I: " << i << std::endl;
        auto sample_0 = sample.index(
                {torch::indexing::TensorIndex(0), torch::indexing::TensorIndex(torch::indexing::Slice()),
                 torch::indexing::TensorIndex(i)});
        // sample_0 with shape [49869]
        auto data = tensor_to_vector(sample_0);
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, 5, 3000);
        auto end = std::chrono::high_resolution_clock::now();
        total_loss += compute_loss(data, means);
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        for (auto &mean: means) {
            std::cout << (float)mean << " ";
        }
        std::cout << "---------------------------------------------" << std::endl;
    }

    // print total time in second
    std::cout << "total time: " << (total_time / (1000)) << "ms" << std::endl;
    std::cout << "total loss: " << total_loss / 768.0 << std::endl;

    return 0;
}
