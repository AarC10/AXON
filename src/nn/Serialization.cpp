#include "nn/Serialization.h"

#include <fstream>
#include <stdexcept>

namespace axon {

void save(const std::vector<Tensor> &params, const std::string &path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    int num_params = static_cast<int>(params.size());
    out.write(reinterpret_cast<const char *>(&num_params), sizeof(num_params));

    for (const auto &param : params) {
        const auto &shape = param->get_shape();
        int ndim = static_cast<int>(shape.size());
        out.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
        out.write(reinterpret_cast<const char *>(shape.data()), ndim * sizeof(int));

        int nelem = param->nelem();
        out.write(reinterpret_cast<const char *>(param->data()), nelem * sizeof(float));
    }
}

void load(std::vector<Tensor> &params, const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    int num_params = 0;
    in.read(reinterpret_cast<char *>(&num_params), sizeof(num_params));
    if (num_params != static_cast<int>(params.size())) {
        throw std::runtime_error("Parameter count mismatch: file has " + std::to_string(num_params) +
                                 " but model has " + std::to_string(params.size()));
    }

    for (auto &param : params) {
        int ndim = 0;
        in.read(reinterpret_cast<char *>(&ndim), sizeof(ndim));

        std::vector<int> shape(ndim);
        in.read(reinterpret_cast<char *>(shape.data()), ndim * sizeof(int));

        if (shape != param->get_shape()) {
            throw std::runtime_error("Shape mismatch when loading parameters");
        }

        int nelem = param->nelem();
        in.read(reinterpret_cast<char *>(param->data()), nelem * sizeof(float));
    }
}

} // namespace axon
