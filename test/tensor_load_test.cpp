//
// Created by Entropy Xu on 7/5/23.
//

#include <iostream>
#include <torch/torch.h>
#include <fstream>
#include <cassert>


std::vector<char> get_the_bytes(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary);

    // Assert file exists
    std::stringstream ss;
    ss << "File not found: " << filename;
    assert(input.is_open() && ss.str().c_str());

    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

int main() {
    std::vector<char> f = get_the_bytes((std::string &) "sample.pt");
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor sample = x.toTensor();

    std::cout << sample.sizes() << std::endl;
}
