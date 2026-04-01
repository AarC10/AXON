#include "core/Tensor.h"

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape(shape),
      require_grad(requires_grad),
      is_leaf(true),
      offset(0) {
    int n = 1;
    for (int d : shape) n *= d;
    storage = std::make_shared<std::vector<float>>(n, 0.0f);
    compute_strides();
}


Tensor::Tensor(std::vector<int> shape, bool require_grad) {}
Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool require_grad) {}
Tensor::Tensor(const Tensor& other) {}
Tensor::Tensor(Tensor&& other) {}
Tensor& Tensor::operator=(const Tensor& other) {}
Tensor& Tensor::operator=(Tensor&& other) {}
Tensor Tensor::zeros(std::vector<int> shape, bool require_grad) {}
Tensor Tensor::ones(std::vector<int> shape, bool require_grad) {}
Tensor Tensor::full(std::vector<int> shape, bool require_grad) {}
Tensor Tensor::randn(std::vector<int> shape, bool require_grad) {}
Tensor Tensor::rand(std::vector<int> shape, bool require_grad) {}
Tensor Tensor::eye(int n, bool require_grad) {}
Tensor Tensor::arange(float start, float stop, float step, bool require_grad) {}
const std::vector<int>& Tensor::get_shape() const {}
const std::vector<int>& Tensor::get_strides() const {}
int Tensor::ndim() const {}
int Tensor::nelem() const {}
int Tensor::size(int dim) const {}
bool Tensor::requires_grad() const {}
bool Tensor::set_requires_grad(bool require_grad) {}
bool Tensor::is_contiguous() {}
float* Tensor::data() {}
const float* Tensor::data() const {}
float Tensor::at(const std::vector<int>& idx) const {}
float& Tensor::at(const std::vector<int>& idx) {}
float Tensor::operator[](int idx) const {}
float& Tensor::operator[](int idx) {}
Tensor Tensor::operator+(const Tensor& rhs) const {}
Tensor Tensor::operator-(const Tensor& rhs) const {}
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

Tensor operator+(float scalar, const Tensor& tensor) {
    return scalar += tensor;
}

Tensor operator-(float scalar, const Tensor& tensor) {
    return scalar -= tensor;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return scalar *= tensor;
}

Tensor operator/(float scalar, const Tensor& tensor) {
    return scalar /= tensor;
}

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
int Tensor::flat_idnex(const std::vector<int>& idx) const {}

void Tensor::compute_strides() {
    strides.resize(shape.size());

    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

std::vector<int> Tensor::broadcast_shape(const std::vector<int>& shape_one, const std::vector<int>& shape_two) {}
