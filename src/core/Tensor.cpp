#include "core/Tensor.h"

#include <random>

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape(shape), require_grad(requires_grad), is_leaf(true), offset(0) {
    int n = 1;
    for (int d : shape) n *= d;
    storage = std::make_shared<std::vector<float>>(n, 0.0f);
    compute_strides();
}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool require_grad)
    : shape(shape), require_grad(require_grad), is_leaf(true), offset(0) {
    int n = 1;
    for (const int d : shape) n *= d;

    if (data.size() != n) {
        throw std::invalid_argument("Data size doesnt match shape");
    }

    storage = std::make_shared<std::vector<float>>(n, 0.0f);
    compute_strides();
}

Tensor::Tensor(const Tensor& other)
    : storage(std::make_shared<std::vector<float>>(*other.storage)), offset(other.offset), shape(other.shape),
      strides(other.strides), require_grad(other.require_grad), is_leaf(other.is_leaf), grad(other.grad),
      inputs(other.inputs) {}

Tensor::Tensor(Tensor&& other)
    : storage(std::move(other.storage)), offset(other.offset), shape(std::move(other.shape)),
      strides(std::move(other.strides)), require_grad(other.require_grad), is_leaf(other.is_leaf),
      grad(std::move(other.grad)), inputs(std::move(other.inputs)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    storage = std::make_shared<std::vector<float>>(*other.storage);
    offset = other.offset;
    shape = other.shape;
    strides = other.strides;
    require_grad = other.require_grad;
    is_leaf = other.is_leaf;
    grad = other.grad;
    inputs = other.inputs;
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) {
    if (this == &other) return *this;
    storage = std::move(other.storage);
    offset = other.offset;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    require_grad = other.require_grad;
    is_leaf = other.is_leaf;
    grad = std::move(other.grad);
    inputs = std::move(other.inputs);
    return *this;
}

Tensor Tensor::zeros(std::vector<int>& shape, bool require_grad) {
    // Zero inited in ctor
    return Tensor(shape, require_grad);
}

Tensor Tensor::ones(std::vector<int>& shape, bool require_grad) {
    return Tensor::full(shape, 1.0f, require_grad);
}

Tensor Tensor::full(std::vector<int>& shape, float value, bool requires_grad) {
    Tensor tensor(shape, requires_grad);
    std::fill(tensor.storage->begin(), tensor.storage->end(), value);
    return tensor;
}

Tensor Tensor::randn(std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    Tensor tensor(shape, require_grad);

    for (float& x : *tensor.storage) {
        x = dis(gen);
    }

    return tensor;
}

Tensor Tensor::rand(std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    Tensor tensor(shape, require_grad);

    for (float& x : *tensor.storage) {
        x = dis(gen);
    }

    return tensor;
}

Tensor Tensor::eye(int n, bool require_grad) {
    Tensor tensor({n, n}, require_grad);
    for (int i = 0; i < n; ++i) {
        (*tensor.storage)[i * n + i] = 1.0f;
    }
    return tensor;
}

Tensor Tensor::arange(float start, float stop, float step, bool require_grad) {
    if (step == 0.0f) {
        throw std::invalid_argument("Step must be non-zero");
    }

    int n = static_cast<int>(std::ceil((stop - start) / step));
    Tensor tensor({n, n}, require_grad);
    for (int i = 0; i < n; ++i) {
        (*tensor.storage)[i] = start + i * step;
    }

    return tensor;
}

const std::vector<int>& Tensor::get_shape() const {
    return shape;
}

const std::vector<int>& Tensor::get_strides() const {
    return strides;
}

int Tensor::ndim() const {
    return static_cast<int>(shape.size());
}

int Tensor::nelem() const {
    int n = 1;
    for (int d : shape) n *= d;
    return n;
}

int Tensor::size(int dim) const {
    // neg index edge case
    if (dim < 0) {
        dim += static_cast<int>(shape.size());
    }

    if (dim < 0 || dim >= shape.size()) {
        throw std::out_of_range("Dimension out of range");
    }

    return shape[dim];
}

bool Tensor::requires_grad() const {
    return require_grad;
}

bool Tensor::set_requires_grad(bool require_grad) {
    this->require_grad = require_grad;
}

bool Tensor::is_contiguous() {
    int stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (strides[i] != stride) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

float* Tensor::data() {
    return storage->data() + offset;
}

const float* Tensor::data() const {
    return storage->data() + offset;
}

float Tensor::at(const std::vector<int>& idx) const {
    int flat_idx = flat_idnex(idx);
    return (*storage)[offset + flat_idx];
}

float& Tensor::at(const std::vector<int>& idx) {
    int flat_idx = flat_idnex(idx);
    return (*storage)[offset + flat_idx];
}

float Tensor::operator[](int idx) const {
    return (*storage)[offset + idx];
}

float& Tensor::operator[](int idx) {
    return (*storage)[offset + idx];
}

Tensor Tensor::operator+(const Tensor& rhs) const {
}

Tensor Tensor::operator-(const Tensor& rhs) const {
}

Tensor Tensor::operator*(const Tensor& rhs) const {}

Tensor Tensor::operator/(const Tensor& rhs) const {}

Tensor Tensor::operator-() const {}

Tensor Tensor::operator+(float scalar) const {}

Tensor Tensor::operator-(float scalar) const {}

Tensor Tensor::operator*(float scalar) const {}

Tensor Tensor::operator/(float scalar) const {}

Tensor& Tensor::operator+=(const Tensor& rhs) {}

Tensor& Tensor::operator-=(const Tensor& rhs) {}

Tensor& Tensor::operator*=(const Tensor& rhs) {}

Tensor& Tensor::operator/=(const Tensor& rhs) {}

Tensor operator+(float scalar, const Tensor& tensor) { return scalar += tensor; }

Tensor operator-(float scalar, const Tensor& tensor) { return scalar -= tensor; }

Tensor operator*(float scalar, const Tensor& tensor) { return scalar *= tensor; }

Tensor operator/(float scalar, const Tensor& tensor) { return scalar /= tensor; }

Tensor Tensor::exp() const {}
Tensor Tensor::log() const {}
Tensor Tensor::sqrt() const {}
Tensor Tensor::abs() const {}
Tensor Tensor::pow(float exponent) const {}
Tensor Tensor::pow(const Tensor& exp) const {}
Tensor Tensor::clip(float min, float max) const {}
Tensor Tensor::operator==(const Tensor& rhs) const {}
Tensor Tensor::operator!=(const Tensor& rhs) const {}
Tensor Tensor::operator<(const Tensor& rhs) const {}
Tensor Tensor::operator<=(const Tensor& rhs) const {}
Tensor Tensor::operator>(const Tensor& rhs) const {}
Tensor Tensor::operator>=(const Tensor& rhs) const {}

int Tensor::flat_idnex(const std::vector<int>& idx) const {
    int flat = offset;

    for (int i = 0; i < idx.size(); i++) {
        flat += idx[i] * strides[i];
    }

    return flat;
}

void Tensor::compute_strides() {
    strides.resize(shape.size());

    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

std::vector<int> Tensor::broadcast_shape(const std::vector<int>& shape_one, const std::vector<int>& shape_two) {
    int ndim = std::max(shape_one.size(), shape_two.size());
    std::vector<int> broadcasted_shape(ndim);

    for (int i = 0; i < ndim; i++) {
        int shape_one_dim = i < shape_one.size() ? shape_one[shape_one.size() - 1 - i] : 1;
        int shape_two_dim = i < shape_two.size() ? shape_two[shape_two.size() - 1 - i] : 1;

        if (shape_one_dim == shape_two_dim) {
            broadcasted_shape[ndim - 1 - i] = shape_one_dim;
        } else if (shape_one_dim > shape_two_dim) {
            broadcasted_shape[ndim - 1 - i] = shape_two_dim;
        } else if (shape_one_dim < shape_two_dim) {
            broadcasted_shape[ndim - 1 - i] = shape_one_dim;
        } else {
            // Shouldnt happen tho
            throw std::out_of_range("Dimension out of range" + std::to_string(shape_one_dim) + " " + std::to_string(shape_two_dim));
        }
    }

    return broadcasted_shape;
}
