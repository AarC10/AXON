#include "core/Tensor.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : offset(0), shape(shape), require_grad(requires_grad), is_leaf(true) {
    int n = 1;
    for (int d : shape) {
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
        throw std::invalid_argument("Data size doesnt match shape");
    }

    storage = std::make_shared<std::vector<float>>(data.begin(), data.end());
    compute_strides();
}

Tensor::Tensor(const Tensor& other)
    : storage(std::make_shared<std::vector<float>>(*other.storage)), offset(other.offset), shape(other.shape),
      strides(other.strides), require_grad(other.require_grad), is_leaf(other.is_leaf), grad(other.grad),
      inputs(other.inputs), gradient_func(other.gradient_func) {}

Tensor::Tensor(Tensor&& other)
    : storage(std::move(other.storage)), offset(other.offset), shape(std::move(other.shape)),
      strides(std::move(other.strides)), require_grad(other.require_grad), is_leaf(other.is_leaf),
      grad(std::move(other.grad)), inputs(std::move(other.inputs)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    storage = std::make_shared<std::vector<float>>(*other.storage);
    offset = other.offset;
    shape = other.shape;
    strides = other.strides;
    require_grad = other.require_grad;
    is_leaf = other.is_leaf;
    grad = other.grad;
    inputs = other.inputs;
    gradient_func = other.gradient_func;
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

    if (dim < 0 || dim >= shape.size()) {
        throw std::out_of_range("Dimension out of range");
    }

    return shape[dim];
}

bool Tensor::requires_grad() const { return require_grad; }

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
    int flat_idx = flat_index(idx);
    return (*storage)[offset + flat_idx];
}

float Tensor::operator[](int idx) const { return (*storage)[offset + idx]; }

float& Tensor::operator[](int idx) { return (*storage)[offset + idx]; }


Tensor Tensor::operator+(const Tensor& rhs) const {
    Tensor out = binary_op(*this, [](const float a, const float b) { return a + b; });

    if (out.require_grad) {
        auto lhs_ptr = std::make_shared<Tensor>(*this);
        auto rhs_ptr = std::make_shared<Tensor>(rhs);

        out.inputs = {lhs_ptr, rhs_ptr};
        out.is_leaf = false;
        out.gradient_func = [lhs_ptr, rhs_ptr](const Tensor& grad) {
            // ADding should pass grad straight through both sides
            if (lhs_ptr->require_grad) {
                lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->grad ? *lhs_ptr->grad + grad : grad);
            }

            if (rhs_ptr->require_grad) {
                rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->grad ? *rhs_ptr->grad + grad : grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a - b; });

    if (out.require_grad) {
        auto lhs_ptr = std::make_shared<Tensor>(*this);
        auto rhs_ptr = std::make_shared<Tensor>(rhs);
        out.inputs = {lhs_ptr, rhs_ptr};
        out.is_leaf = false;
        out.gradient_func = [lhs_ptr, rhs_ptr](const Tensor& grad) {
            // lhs should get +grad
            if (lhs_ptr->require_grad) {
                lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->grad ? *lhs_ptr->grad + grad : grad);
            }

            // rhs gets -grad
            if (rhs_ptr->require_grad) {
                Tensor neg_grad = -grad;
                rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->grad ? *rhs_ptr->grad + neg_grad : neg_grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator*(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a * b; });

    if (out.require_grad) {
        auto lhs_ptr = std::make_shared<Tensor>(*this);
        auto rhs_ptr = std::make_shared<Tensor>(rhs);

        out.inputs = {lhs_ptr, rhs_ptr};
        out.is_leaf = false;
        out.gradient_func = [lhs_ptr, rhs_ptr](const Tensor& grad) {
            // d/da (a*b) = b
            // d/db (a*b) = a
            if (lhs_ptr->require_grad) {
                Tensor lhs_grad = grad * *rhs_ptr;
                lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->grad ? *lhs_ptr->grad + lhs_grad : lhs_grad);
            }
            if (rhs_ptr->require_grad) {
                Tensor rhs_grad = grad * *lhs_ptr;
                rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->grad ? *rhs_ptr->grad + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    Tensor out = binary_op(rhs, [](const float a, const float b) { return a / b; });

    if (out.require_grad) {
        auto lhs_ptr = std::make_shared<Tensor>(*this);
        auto rhs_ptr = std::make_shared<Tensor>(rhs);
        out.inputs = {lhs_ptr, rhs_ptr};
        out.is_leaf = false;
        out.gradient_func = [lhs_ptr, rhs_ptr](const Tensor& grad) {
            // d/da (a/b) = 1/b
            // d/db (a/b) = -a/b^2
            if (lhs_ptr->require_grad) {
                Tensor lhs_grad = grad / *rhs_ptr;
                lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->grad ? *lhs_ptr->grad + lhs_grad : lhs_grad);
            }
            if (rhs_ptr->require_grad) {
                Tensor rhs_grad = grad * (*lhs_ptr * -1.0f) / (*rhs_ptr * *rhs_ptr);
                rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->grad ? *rhs_ptr->grad + rhs_grad : rhs_grad);
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
    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] += (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] -= (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor& Tensor::operator*=(const Tensor& rhs) {
    for (int i = 0; i < nelem(); ++i) {
        (*storage)[offset + i] *= (*rhs.storage)[rhs.offset + i];
    }

    return *this;
}

Tensor& Tensor::operator/=(const Tensor& rhs) {
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
        (*out.storage)[i] = std::exp((*out.storage)[i + offset]);
    }

    if (require_grad) {
        auto self = std::make_shared<Tensor>(*this);
        auto out_ptr = std::make_shared<Tensor>(out);

        out.inputs = {self};
        out.is_leaf = false;
        out.gradient_func = [self, out_ptr](const Tensor& grad) {
            // d/dx exp(x) = exp(x)
            Tensor lhs_grad = grad * *out_ptr;
            self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::log() const {
    Tensor out(shape, require_grad);

    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::log((*out.storage)[i + offset]);
    }

    if (require_grad) {
        auto self = std::make_shared<Tensor>(*this);
        out.inputs = {self};
        out.is_leaf = false;
        out.gradient_func = [self](const Tensor& grad) {
            // d/dx log(x) = 1/x
            Tensor lhs_grad = grad / *self;
            self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}
Tensor Tensor::sqrt() const {
    Tensor out(shape, require_grad);
    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::sqrt((*out.storage)[i + offset]);
    }

    if (require_grad) {
        auto self = std::make_shared<Tensor>(*this);
        auto out_ptr = std::make_shared<Tensor>(out);

        out.inputs = {self};
        out.is_leaf = false;
        out.gradient_func = [self, out_ptr](const Tensor& grad) {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x))
            Tensor lhs_grad = grad / (*out_ptr * 2.0f);
            self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
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
        auto self = std::make_shared<Tensor>(*this);
        out.inputs = {self};
        out.is_leaf = false;
        out.gradient_func = [self](const Tensor& grad) {
            // d/dx abs(x) = sign(x)
            Tensor sign(self->shape, false);
            for (int i = 0; i < self->nelem(); i++) {
                float val = (*self->storage)[self->offset + i];
                (*sign.storage)[i] = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
            }
            Tensor lhs_grad = grad * sign;
            self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
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
        auto self = std::make_shared<Tensor>(*this);

        out.inputs = {self};
        out.is_leaf = false;
        out.gradient_func = [self, exponent](const Tensor& grad) {
            // d/dx x^n = n * x^(n-1)
            Tensor lhs_grad = grad * self->pow(exponent - 1.0f) * exponent;
            self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::pow(const Tensor& exp) const {
    Tensor out = binary_op(exp, [](const float a, const float b) { return std::pow(a, b); });

    if (out.require_grad) {
        auto self = std::make_shared<Tensor>(*this);
        auto exp_ptr = std::make_shared<Tensor>(exp);
        out.inputs = {self, exp_ptr};
        out.is_leaf = false;
        out.gradient_func = [self, exp_ptr](const Tensor& grad) {
            // d/da a^b = b * a^(b-1)
            if (self->require_grad) {
                Tensor lhs_grad = grad * *exp_ptr * self->pow(*exp_ptr - 1.0f);
                self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
            }
            // d/db a^b = a^b * log(a)
            if (exp_ptr->require_grad) {
                Tensor rhs_grad = grad * self->pow(*exp_ptr) * self->log();
                exp_ptr->grad = std::make_shared<Tensor>(exp_ptr->grad ? *exp_ptr->grad + rhs_grad : rhs_grad);
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
        auto self = std::make_shared<Tensor>(*this);
        out.inputs = {self};
        out.is_leaf = false;
        out.gradient_func = [self, min, max](const Tensor& grad) {
            // gradient is 1 where input was in [min, max], 0 where it was clipped
            Tensor mask(self->shape, false);
            for (int i = 0; i < self->nelem(); i++) {
                float val = (*self->storage)[self->offset + i];
                (*mask.storage)[i] = (val > min && val < max) ? 1.0f : 0.0f;
            }
            Tensor lhs_grad = grad * mask;
            self->grad = std::make_shared<Tensor>(self->grad ? *self->grad + lhs_grad : lhs_grad);
        };
    }

    return out;
}

Tensor Tensor::operator==(const Tensor& rhs) const {
    return binary_op(rhs, [](const float a, const float b) { return a == b ? 1.0f : 0.0f; });
}

Tensor Tensor::operator!=(const Tensor& rhs) const {
    return binary_op(rhs, [](const float a, const float b) { return a != b ? 1.0f : 0.0f; });
}

Tensor Tensor::operator<(const Tensor& rhs) const {
    return binary_op(rhs, [](const float a, const float b) { return a < b ? 1.0f : 0.0f; });
}

Tensor Tensor::operator<=(const Tensor& rhs) const {
    return binary_op(rhs, [](const float a, const float b) { return a <= b ? 1.0f : 0.0f; });
}

Tensor Tensor::operator>(const Tensor& rhs) const {
    return binary_op(rhs, [](const float a, const float b) { return a > b ? 1.0f : 0.0f; });
}

Tensor Tensor::operator>=(const Tensor& rhs) const {
    return binary_op(rhs, [](const float a, const float b) { return a >= b ? 1.0f : 0.0f; });
}

void Tensor::set_gradient_func(GradientFunc func, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    this->gradient_func = func;
    this->inputs = inputs;
}

bool Tensor::get_is_leaf() const { return is_leaf; }

int Tensor::flat_index(const std::vector<int>& idx) const {
    int flat = offset;

    if (flat < 0 || flat >= nelem()) {
        throw std::out_of_range("Flat index out of range");
    }

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

        if (shape_one_dim == shape_two_dim) {
            broadcasted_shape[ndim - 1 - i] = shape_one_dim;
        } else if (shape_one_dim > shape_two_dim) {
            broadcasted_shape[ndim - 1 - i] = shape_two_dim;
        } else if (shape_one_dim < shape_two_dim) {
            broadcasted_shape[ndim - 1 - i] = shape_one_dim;
        } else {
            // Shouldnt happen tho
            throw std::out_of_range("Dimension out of range" + std::to_string(shape_one_dim) + " " +
                                    std::to_string(shape_two_dim));
        }
    }

    return broadcasted_shape;
}
