#include <iostream>
#include <torch/torch.h>
#include "constants.h"
#include "kmeans.h"

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

std::vector<float_type> tensor_to_vector(torch::Tensor tensor) {
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

int main() {
    std::vector<char> f = get_the_bytes("sample.pt");
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor sample = x.toTensor();

    long total_time = 0;
    for (int i = 0; i < 768; i++) {
        auto sample_0 = sample.index(
                {torch::indexing::TensorIndex(0), torch::indexing::TensorIndex(torch::indexing::Slice()),
                 torch::indexing::TensorIndex(i)});
        // sample_0 with shape [49869]
        auto data = tensor_to_vector(sample_0);
        auto start = std::chrono::high_resolution_clock::now();
        auto means = kmeans(data, 4, 3000);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        for (auto &mean: means) {
            std::cout << (float)mean << " ";
        }
        std::cout << std::endl;
        std::cout << means.size() << std::endl;
    }

    // print total time in second
    std::cout << "total time: " << (total_time / (1000)) << "ms" << std::endl;

    return 0;
}
