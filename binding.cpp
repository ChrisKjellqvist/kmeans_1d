//
// Created by Entropy Xu on 7/2/23.
//

#include "binding.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>

at::Tensor increase_by_one(at::Tensor x) {
    x = x.add_(1.0);
    return x;
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("increase_by_one", &increase_by_one, "A function that increments each float value in the tensor by one");
}
