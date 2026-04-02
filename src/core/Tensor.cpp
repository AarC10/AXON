#include "core/Tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : offset(0), shape(shape), require_grad(requires_grad), is_leaf(true) {
    int n = 1;
    for (int d : shape) {
        if (d <= 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }

        n *= d;
    }
    storage = std::make_shared<std::vector<float>>(n, 0.0f);
    compute_strides();
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool require_grad)
    : offset(0), shape(shape), require_grad(require_grad), is_leaf(true) {
    int n = 1;
    for (const int d : shape) {
        n *= d;
    }

    if (data.size() != n) {
        throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
            ") doesn't match tensor shape (expected " + std::to_string(n) + " elements)");
    }

    storage = std::make_shared<std::vector<float>>(data.begin(), data.end());
    compute_strides();
}

Tensor::Tensor(const Tensor& other)
    : storage(std::make_shared<std::vector<float>>(*other.storage)), offset(other.offset), shape(other.shape),
      strides(other.strides), require_grad(false), is_leaf(true) {}

Tensor::Tensor(Tensor&& other)
    : storage(std::move(other.storage)), offset(other.offset), shape(std::move(other.shape)),
      strides(std::move(other.strides)), require_grad(other.require_grad), is_leaf(other.is_leaf),
      grad(std::move(other.grad)), inputs(std::move(other.inputs)), gradient_func(std::move(other.gradient_func)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    storage = std::make_shared<std::vector<float>>(*other.storage);
    offset = other.offset;
    shape = other.shape;
    strides = other.strides;
    require_grad = false;
    is_leaf = true;
    grad = nullptr;
    inputs.clear();
    gradient_func = nullptr;
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) {
    if (this == &other) {
        return *this;
    }
    storage = std::move(other.storage);
    offset = other.offset;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    require_grad = other.require_grad;
    is_leaf = other.is_leaf;
    grad = std::move(other.grad);
    inputs = std::move(other.inputs);
    gradient_func = std::move(other.gradient_func);
    return *this;
}

Tensor Tensor::zeros(const std::vector<int>& shape, bool require_grad) {
    // Zero inited in ctor
    return Tensor(shape, require_grad);
}

Tensor Tensor::ones(const std::vector<int>& shape, bool require_grad) {
    return Tensor::full(shape, 1.0f, require_grad);
}

Tensor Tensor::full(const std::vector<int>& shape, float value, bool requires_grad) {
    Tensor tensor(shape, requires_grad);
    std::fill(tensor.storage->begin(), tensor.storage->end(), value);
    return tensor;
}

Tensor Tensor::randn(const std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    Tensor tensor(shape, require_grad);

    for (float& x : *tensor.storage) {
        x = dis(gen);
    }

    return tensor;
}

Tensor Tensor::rand(const std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

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
    if (n < 0) {
        throw std::invalid_argument("Invalid range: start=" + std::to_string(start) + ", stop=" + std::to_string(stop) +
            "), step=" + std::to_string(step));
    }

    Tensor tensor({n}, require_grad);
    for (int i = 0; i < n; ++i) {
        (*tensor.storage)[i] = start + i * step;
    }

    return tensor;
}

const std::vector<int>& Tensor::get_shape() const { return shape; }

const std::vector<int>& Tensor::get_strides() const { return strides; }

int Tensor::ndim() const { return static_cast<int>(shape.size()); }

int Tensor::nelem() const {
    int n = 1;
    for (int d : shape) {
        n *= d;
    }
    return n;
}

int Tensor::size(int dim) const {
    // neg index edge case
    if (dim < 0) {
        dim += static_cast<int>(shape.size());
    }

    if (dim < 0 || dim >= static_cast<int>(shape.size())) {
        throw std::out_of_range("Dimension out of range");
    }

    return shape[dim];
}

bool Tensor::get_require_grad() const { return require_grad; }

void Tensor::set_require_grad(const bool require_grad) { this->require_grad = require_grad; }

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

float* Tensor::data() { return storage->data() + offset; }

const float* Tensor::data() const { return storage->data() + offset; }

float Tensor::at(const std::vector<int>& idx) const {
    return (*storage)[flat_index(idx)];
}

float& Tensor::at(const std::vector<int>& idx) {
    return (*storage)[flat_index(idx)];
}

float Tensor::operator[](int idx) const { return (*storage)[offset + idx]; }

float& Tensor::operator[](int idx) { return (*storage)[offset + idx]; }


Tensor Tensor::operator+(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a + b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<Tensor>(*this);
        auto rhs_data = std::make_shared<Tensor>(rhs);

        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const Tensor& grad) {
            // Adding should pass grad straight through both sides
            if (lhs_data->require_grad) {
                lhs_data->grad = std::make_shared<Tensor>(lhs_data->grad ? *lhs_data->grad + grad : grad);
            }

            if (rhs_data->require_grad) {
                rhs_data->grad = std::make_shared<Tensor>(rhs_data->grad ? *rhs_data->grad + grad : grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a - b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<Tensor>(*this);
        auto rhs_data = std::make_shared<Tensor>(rhs);
        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const Tensor& grad) {
            // lhs should get +grad
            if (lhs_data->require_grad) {
                lhs_data->grad = std::make_shared<Tensor>(lhs_data->grad ? *lhs_data->grad + grad : grad);
            }

            // rhs gets -grad
            if (rhs_data->require_grad) {
                Tensor neg_grad = -grad;
                rhs_data->grad = std::make_shared<Tensor>(rhs_data->grad ? *rhs_data->grad + neg_grad : neg_grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator*(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a * b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<Tensor>(*this);
        auto rhs_data = std::make_shared<Tensor>(rhs);

        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const Tensor& grad) {
            // d/da (a*b) = b
            // d/db (a*b) = a
            if (lhs_data->require_grad) {
                Tensor lhs_grad = grad * *rhs_data;
                lhs_data->grad = std::make_shared<Tensor>(lhs_data->grad ? *lhs_data->grad + lhs_grad : lhs_grad);
            }
            if (rhs_data->require_grad) {
                Tensor rhs_grad = grad * *lhs_data;
                rhs_data->grad = std::make_shared<Tensor>(rhs_data->grad ? *rhs_data->grad + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a / b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<Tensor>(*this);
        auto rhs_data = std::make_shared<Tensor>(rhs);
        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const Tensor& grad) {
            // d/da (a/b) = 1/b
            // d/db (a/b) = -a/b^2
            if (lhs_data->require_grad) {
                Tensor lhs_grad = grad / *rhs_data;
                lhs_data->grad = std::make_shared<Tensor>(lhs_data->grad ? *lhs_data->grad + lhs_grad : lhs_grad);
            }
            if (rhs_data->require_grad) {
                Tensor rhs_grad = grad * (*lhs_data * -1.0f) / (*rhs_data * *rhs_data);
                rhs_data->grad = std::make_shared<Tensor>(rhs_data->grad ? *rhs_data->grad + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator-() const { return *this * -1.0f; }

Tensor Tensor::operator+(float scalar) const { return *this + full(shape, scalar); }

Tensor Tensor::operator-(float scalar) const { return *this - full(shape, scalar); }

Tensor Tensor::operator*(float scalar) const { return *this * full(shape, scalar); }

Tensor Tensor::operator/(float scalar) const { return *this / full(shape, scalar); }

Tensor& Tensor::operator+=(const Tensor& rhs) {
    if (require_grad && !is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (shape != rhs.shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] += (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
    if (require_grad && !is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (shape != rhs.shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }


    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] -= (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor& Tensor::operator*=(const Tensor& rhs) {
    if (require_grad && !is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (shape != rhs.shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] *= (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor& Tensor::operator/=(const Tensor& rhs) {
    if (require_grad && !is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (shape != rhs.shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] /= (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor operator+(float scalar, const Tensor& tensor) { return tensor + scalar; }

Tensor operator-(float scalar, const Tensor& tensor) { return Tensor::full(tensor.shape, scalar) - tensor; }

Tensor operator*(float scalar, const Tensor& tensor) { return tensor * scalar; }

Tensor operator/(float scalar, const Tensor& tensor) { return Tensor::full(tensor.shape, scalar) / tensor; }

Tensor Tensor::exp() const {
    Tensor out(shape, require_grad);

    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::exp((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);
        auto out_ptr = std::make_shared<Tensor>(out);

        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, out_ptr](const Tensor& grad) {
            // d/dx exp(x) = exp(x)
            Tensor lhs_grad = grad * *out_ptr;
            self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::log() const {
    Tensor out(shape, require_grad);

    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::log((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);
        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data](const Tensor& grad) {
            // d/dx log(x) = 1/x
            Tensor lhs_grad = grad / *self_data;
            self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}
Tensor Tensor::sqrt() const {
    Tensor out(shape, require_grad);
    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::sqrt((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);
        auto out_ptr = std::make_shared<Tensor>(out);

        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, out_ptr](const Tensor& grad) {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x))
            Tensor lhs_grad = grad / (*out_ptr * 2.0f);
            self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}
Tensor Tensor::abs() const {
    Tensor out(shape, require_grad);
    for (int i = 0; i < nelem(); i++) {
        (*out.storage)[i] = std::abs((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);
        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data](const Tensor& grad) {
            // d/dx abs(x) = sign(x)
            Tensor sign(self_data->shape, false);
            for (int i = 0; i < self_data->nelem(); i++) {
                float val = (*self_data->storage)[self_data->offset + i];
                (*sign.storage)[i] = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
            }
            Tensor lhs_grad = grad * sign;
            self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::pow(float exponent) const {
    Tensor out(shape, require_grad);
    for (int i = 0; i < nelem(); i++) {
        (*out.storage)[i] = std::pow((*storage)[offset + i], exponent);
    }

    if (require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);

        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, exponent](const Tensor& grad) {
            // d/dx x^n = n * x^(n-1)
            Tensor lhs_grad = grad * self_data->pow(exponent - 1.0f) * exponent;
            self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::pow(const Tensor& exp) const {
    Tensor out = binary_op(exp, [](const float a, const float b) { return std::pow(a, b); });

    if (out.require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);
        auto exp_data = std::make_shared<Tensor>(exp);
        out.inputs = {self_data, exp_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, exp_data](const Tensor& grad) {
            // d/da a^b = b * a^(b-1)
            if (self_data->require_grad) {
                Tensor lhs_grad = grad * *exp_data * self_data->pow(*exp_data - 1.0f);
                self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
            }
            // d/db a^b = a^b * log(a)
            if (exp_data->require_grad) {
                Tensor rhs_grad = grad * self_data->pow(*exp_data) * self_data->log();
                exp_data->grad = std::make_shared<Tensor>(exp_data->grad ? *exp_data->grad + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

Tensor Tensor::clip(float min, float max) const {
    Tensor out(shape, require_grad);
    for (int i = 0; i < nelem(); i++) {
        (*out.storage)[i] = std::clamp((*storage)[offset + i], min, max);
    }

    if (require_grad) {
        auto self_data = std::make_shared<Tensor>(*this);
        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, min, max](const Tensor& grad) {
            // gradient is 1 where input was in [min, max], 0 where it was clipped
            Tensor mask(self_data->shape, false);
            for (int i = 0; i < self_data->nelem(); i++) {
                float val = (*self_data->storage)[self_data->offset + i];
                (*mask.storage)[i] = (val >= min && val <= max) ? 1.0f : 0.0f;
            }
            Tensor lhs_grad = grad * mask;
            self_data->grad = std::make_shared<Tensor>(self_data->grad ? *self_data->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::operator==(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a == b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

Tensor Tensor::operator!=(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a != b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

Tensor Tensor::operator<(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a < b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

Tensor Tensor::operator<=(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a <= b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

Tensor Tensor::operator>(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a > b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

Tensor Tensor::operator>=(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a >= b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

void Tensor::set_gradient_func(GradientFunc func, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    this->gradient_func = func;
    this->inputs = inputs;
}

bool Tensor::get_is_leaf() const { return is_leaf; }

int Tensor::flat_index(const std::vector<int>& idx) const {
    if (idx.size() != shape.size()) {
        throw std::invalid_argument("Index dimensionality does not match tensor shape");
    }

    int flat = offset;

    for (std::size_t i = 0; i < idx.size(); ++i) {
        const int index = idx[i];

        if (index < 0 || index >= shape[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        flat += index * strides[i];
    }

    return flat;
}

void Tensor::detach_grad_state() {
    require_grad = false;
    is_leaf = true;
    inputs.clear();
    gradient_func = nullptr;
}

void Tensor::compute_strides() {
    strides.resize(shape.size());

    int stride = 1;

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

int Tensor::broadcast_index(int flat, const std::vector<int>& out_shape, const std::vector<int>& in_shape) const {
    int ndim = out_shape.size();
    int in_offset = 0;
    int stride = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int out_idx = flat % out_shape[i];
        flat /= out_shape[i];

        int in_dim = (i >= ndim - in_shape.size()) ? in_shape[i - (ndim - in_shape.size())] : 1;
        int in_idx = (in_dim == 1) ? 0 : out_idx;

        in_offset += stride * in_idx;
        stride *= in_dim;
    }

    return in_offset;
}

std::vector<int> Tensor::broadcast_shape(const std::vector<int>& shape_one, const std::vector<int>& shape_two) {
    int ndim = std::max(shape_one.size(), shape_two.size());
    std::vector<int> broadcasted_shape(ndim);

    for (int i = 0; i < ndim; i++) {
        int shape_one_dim = i < shape_one.size() ? shape_one[shape_one.size() - 1 - i] : 1;
        int shape_two_dim = i < shape_two.size() ? shape_two[shape_two.size() - 1 - i] : 1;

        if (shape_one_dim == shape_two_dim || shape_one_dim == 1 || shape_two_dim == 1) {
            broadcasted_shape[ndim - 1 - i] = std::max(shape_one_dim, shape_two_dim);
        } else {
            throw std::invalid_argument(
                "Incompatible dimensions for broadcasting: " + std::to_string(shape_one_dim) + " and " +
                std::to_string(shape_two_dim));
        }
    }

    return broadcasted_shape;
}
