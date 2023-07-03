#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "kmeans.h"

at::Tensor increase_by_one(at::Tensor x) {
    x = x.add_(1.0);
    return x;
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

at::Tensor kmeans_wrapper(at::Tensor x, int k, size_t max_iterations) {
    std::vector<float_type> vec = tensor_to_vector(x);
    std::vector<double> centroids = kmeans(vec, k, max_iterations);
    return torch::tensor(centroids);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("increase_by_one", &increase_by_one, "A function that increments each float value in the tensor by one");
    m.def("kmeans", &kmeans_wrapper, "A kmeans function that takes tensor, k, max_iterations as input and returns tensor centroids");
}
