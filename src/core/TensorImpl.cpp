#include "core/TensorImpl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <iostream>

TensorImpl::TensorImpl(const std::vector<int>& shape, bool requires_grad)
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

TensorImpl::TensorImpl(const std::vector<float>& data, const std::vector<int>& shape, bool require_grad)
    : offset(0), shape(shape), require_grad(require_grad), is_leaf(true) {
    int n = 1;
    for (const int d : shape) {
        if (d <= 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }

        n *= d;
    }

    if (data.size() != n) {
        throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                    ") doesn't match tensor shape (expected " + std::to_string(n) + " elements)");
    }

    storage = std::make_shared<std::vector<float>>(data.begin(), data.end());
    compute_strides();
}

TensorImpl::TensorImpl(const TensorImpl& other)
    : storage(std::make_shared<std::vector<float>>(*other.storage)), offset(other.offset), shape(other.shape),
      strides(other.strides), require_grad(other.require_grad), is_leaf(other.is_leaf), gradient(other.gradient),
      inputs(other.inputs), gradient_func(other.gradient_func), gradient_func_ptr(other.gradient_func_ptr) {}

TensorImpl::TensorImpl(TensorImpl&& other)
    : storage(std::move(other.storage)), offset(other.offset), shape(std::move(other.shape)),
      strides(std::move(other.strides)), require_grad(other.require_grad), is_leaf(other.is_leaf),
      gradient(std::move(other.gradient)), inputs(std::move(other.inputs)), gradient_func(std::move(other.gradient_func)), gradient_func_ptr(std::move(other.gradient_func_ptr)) {}

TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
    if (this == &other) {
        return *this;
    }
    storage = std::make_shared<std::vector<float>>(*other.storage);
    offset = other.offset;
    shape = other.shape;
    strides = other.strides;
    require_grad = other.require_grad;
    is_leaf = other.is_leaf;
    gradient = other.gradient;
    inputs = other.inputs;
    gradient_func = other.gradient_func;
    return *this;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) {
    if (this == &other) {
        return *this;
    }
    storage = std::move(other.storage);
    offset = other.offset;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    require_grad = other.require_grad;
    is_leaf = other.is_leaf;
    gradient = std::move(other.gradient);
    inputs = std::move(other.inputs);
    gradient_func = std::move(other.gradient_func);
    return *this;
}

TensorImpl TensorImpl::zeros(const std::vector<int>& shape, bool require_grad) {
    // Zero inited in ctor
    return TensorImpl(shape, require_grad);
}

TensorImpl TensorImpl::ones(const std::vector<int>& shape, bool require_grad) {
    return TensorImpl::full(shape, 1.0f, require_grad);
}

TensorImpl TensorImpl::full(const std::vector<int>& shape, float value, bool requires_grad) {
    TensorImpl tensor(shape, requires_grad);
    std::fill(tensor.storage->begin(), tensor.storage->end(), value);
    return tensor;
}

TensorImpl TensorImpl::randn(const std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    TensorImpl tensor(shape, require_grad);

    for (float& x : *tensor.storage) {
        x = dis(gen);
    }

    return tensor;
}

TensorImpl TensorImpl::rand(const std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    TensorImpl tensor(shape, require_grad);

    for (float& x : *tensor.storage) {
        x = dis(gen);
    }

    return tensor;
}

TensorImpl TensorImpl::eye(int n, bool require_grad) {
    TensorImpl tensor({n, n}, require_grad);
    for (int i = 0; i < n; ++i) {
        (*tensor.storage)[i * n + i] = 1.0f;
    }
    return tensor;
}

TensorImpl TensorImpl::arange(float start, float stop, float step, bool require_grad) {
    if (step == 0.0f) {
        throw std::invalid_argument("Step must be non-zero");
    }

    int n = static_cast<int>(std::ceil((stop - start) / step));

    if (n <= 0) {
        throw std::invalid_argument("Invalid range: start=" + std::to_string(start) + ", stop=" + std::to_string(stop) +
                                    "), step=" + std::to_string(step));
    }

    TensorImpl tensor({n}, require_grad);
    for (int i = 0; i < n; ++i) {
        (*tensor.storage)[i] = start + i * step;
    }

    return tensor;
}

const std::vector<int>& TensorImpl::get_shape() const { return shape; }

const std::vector<int>& TensorImpl::get_strides() const { return strides; }

int TensorImpl::ndim() const { return static_cast<int>(shape.size()); }

int TensorImpl::nelem() const {
    int n = 1;
    for (int d : shape) {
        n *= d;
    }
    return n;
}

int TensorImpl::size(int dim) const {
    // neg index edge case
    if (dim < 0) {
        dim += static_cast<int>(shape.size());
    }

    if (dim < 0 || dim >= static_cast<int>(shape.size())) {
        throw std::out_of_range("Dimension out of range");
    }

    return shape[dim];
}

bool TensorImpl::get_require_grad() const { return require_grad; }

void TensorImpl::set_require_grad(const bool require_grad) { this->require_grad = require_grad; }

bool TensorImpl::is_contiguous() const {
    int stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (strides[i] != stride) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

float* TensorImpl::data() { return storage->data() + offset; }

const float* TensorImpl::data() const { return storage->data() + offset; }

float TensorImpl::at(const std::vector<int>& idx) const { return (*storage)[flat_index(idx)]; }

float& TensorImpl::at(const std::vector<int>& idx) { return (*storage)[flat_index(idx)]; }

float TensorImpl::operator[](int idx) const { return (*storage)[offset + idx]; }

float& TensorImpl::operator[](int idx) { return (*storage)[offset + idx]; }

TensorImpl TensorImpl::operator+(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a + b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<TensorImpl>(*this);
        auto rhs_data = std::make_shared<TensorImpl>(rhs);

        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const TensorImpl& grad) {
            // Adding should pass grad straight through both sides
            if (lhs_data->require_grad) {
                lhs_data->gradient = std::make_shared<TensorImpl>(lhs_data->gradient ? *lhs_data->gradient + grad : grad);
            }

            if (rhs_data->require_grad) {
                rhs_data->gradient = std::make_shared<TensorImpl>(rhs_data->gradient ? *rhs_data->gradient + grad : grad);
            }
        };
    }

    return out;
}

Tensor operator+(Tensor lhs_data, Tensor rhs_data) {
    TensorImpl out = lhs_data->binary_op(*rhs_data, [](const float a, const float b) { return a + b; });

    if (out.require_grad) {
        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func_ptr = [lhs_data, rhs_data](std::shared_ptr<TensorImpl> grad) {
            // Adding should pass grad straight through both sides
            if (lhs_data->require_grad) {
                lhs_data->gradient = lhs_data->gradient ? lhs_data->gradient + grad : std::make_shared<TensorImpl>(*grad);
            }

            if (rhs_data->require_grad) {
                rhs_data->gradient = rhs_data->gradient ? rhs_data->gradient + grad : std::make_shared<TensorImpl>(*grad);
            }
        };
    }

    return std::make_shared<TensorImpl>(out);
}

TensorImpl TensorImpl::operator-(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a - b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<TensorImpl>(*this);
        auto rhs_data = std::make_shared<TensorImpl>(rhs);
        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const TensorImpl& grad) {
            // lhs should get +grad
            if (lhs_data->require_grad) {
                lhs_data->gradient = std::make_shared<TensorImpl>(lhs_data->gradient ? *lhs_data->gradient + grad : grad);
            }

            // rhs gets -grad
            if (rhs_data->require_grad) {
                TensorImpl neg_grad = -grad;
                rhs_data->gradient = std::make_shared<TensorImpl>(rhs_data->gradient ? *rhs_data->gradient + neg_grad : neg_grad);
            }
        };
    }

    return out;
}

TensorImpl TensorImpl::operator*(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a * b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<TensorImpl>(*this);
        auto rhs_data = std::make_shared<TensorImpl>(rhs);

        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const TensorImpl& grad) {
            // d/da (a*b) = b
            // d/db (a*b) = a
            if (lhs_data->require_grad) {
                TensorImpl lhs_grad = grad * *rhs_data;
                lhs_data->gradient = std::make_shared<TensorImpl>(lhs_data->gradient ? *lhs_data->gradient + lhs_grad : lhs_grad);
            }
            if (rhs_data->require_grad) {
                TensorImpl rhs_grad = grad * *lhs_data;
                rhs_data->gradient = std::make_shared<TensorImpl>(rhs_data->gradient ? *rhs_data->gradient + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

TensorImpl TensorImpl::operator/(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a / b; });

    if (out.require_grad) {
        auto lhs_data = std::make_shared<TensorImpl>(*this);
        auto rhs_data = std::make_shared<TensorImpl>(rhs);
        out.inputs = {lhs_data, rhs_data};
        out.is_leaf = false;
        out.gradient_func = [lhs_data, rhs_data](const TensorImpl& grad) {
            // d/da (a/b) = 1/b
            // d/db (a/b) = -a/b^2
            if (lhs_data->require_grad) {
                TensorImpl lhs_grad = grad / *rhs_data;
                lhs_data->gradient = std::make_shared<TensorImpl>(lhs_data->gradient ? *lhs_data->gradient + lhs_grad : lhs_grad);
            }
            if (rhs_data->require_grad) {
                TensorImpl rhs_grad = grad * (*lhs_data * -1.0f) / (*rhs_data * *rhs_data);
                rhs_data->gradient = std::make_shared<TensorImpl>(rhs_data->gradient ? *rhs_data->gradient + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

TensorImpl TensorImpl::operator-() const { return *this * -1.0f; }

TensorImpl TensorImpl::operator+(float scalar) const { return *this + full(shape, scalar); }

TensorImpl TensorImpl::operator-(float scalar) const { return *this - full(shape, scalar); }

TensorImpl TensorImpl::operator*(float scalar) const { return *this * full(shape, scalar); }

TensorImpl TensorImpl::operator/(float scalar) const { return *this / full(shape, scalar); }

TensorImpl& TensorImpl::operator+=(const TensorImpl& rhs) {
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

TensorImpl& TensorImpl::operator-=(const TensorImpl& rhs) {
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

TensorImpl& TensorImpl::operator*=(const TensorImpl& rhs) {
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

TensorImpl& TensorImpl::operator/=(const TensorImpl& rhs) {
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

TensorImpl operator+(float scalar, const TensorImpl& tensor) { return tensor + scalar; }

TensorImpl operator-(float scalar, const TensorImpl& tensor) { return TensorImpl::full(tensor.shape, scalar) - tensor; }

TensorImpl operator*(float scalar, const TensorImpl& tensor) { return tensor * scalar; }

TensorImpl operator/(float scalar, const TensorImpl& tensor) { return TensorImpl::full(tensor.shape, scalar) / tensor; }

TensorImpl TensorImpl::exp() const {
    TensorImpl out(shape, require_grad);

    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::exp((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);
        auto out_ptr = std::make_shared<TensorImpl>(out);

        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data](const TensorImpl& grad) {
            // d/dx exp(x) = exp(x)
            TensorImpl lhs_grad = grad * self_data->exp();
            self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
        };
    }

    return out;
}

TensorImpl TensorImpl::log() const {
    TensorImpl out(shape, require_grad);

    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::log((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);
        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data](const TensorImpl& grad) {
            // d/dx log(x) = 1/x
            TensorImpl lhs_grad = grad / *self_data;
            self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
        };
    }

    return out;
}
TensorImpl TensorImpl::sqrt() const {
    TensorImpl out(shape, require_grad);
    for (int i = 0; i < nelem(); ++i) {
        (*out.storage)[i] = std::sqrt((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);
        auto out_ptr = std::make_shared<TensorImpl>(out);

        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, out_ptr](const TensorImpl& grad) {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x))
            TensorImpl lhs_grad = grad / (*out_ptr * 2.0f);
            self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
        };
    }

    return out;
}
TensorImpl TensorImpl::abs() const {
    TensorImpl out(shape, require_grad);
    for (int i = 0; i < nelem(); i++) {
        (*out.storage)[i] = std::abs((*storage)[offset + i]);
    }

    if (require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);
        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data](const TensorImpl& grad) {
            // d/dx abs(x) = sign(x)
            TensorImpl sign(self_data->shape, false);
            for (int i = 0; i < self_data->nelem(); i++) {
                float val = (*self_data->storage)[self_data->offset + i];
                (*sign.storage)[i] = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
            }
            TensorImpl lhs_grad = grad * sign;
            self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
        };
    }

    return out;
}

TensorImpl TensorImpl::pow(float exponent) const {
    TensorImpl out(shape, require_grad);
    for (int i = 0; i < nelem(); i++) {
        (*out.storage)[i] = std::pow((*storage)[offset + i], exponent);
    }

    if (require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);

        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, exponent](const TensorImpl& grad) {
            // d/dx x^n = n * x^(n-1)
            TensorImpl lhs_grad = grad * self_data->pow(exponent - 1.0f) * exponent;
            self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
        };
    }

    return out;
}

TensorImpl TensorImpl::pow(const TensorImpl& exp) const {
    TensorImpl out = binary_op(exp, [](const float a, const float b) { return std::pow(a, b); });

    if (out.require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);
        auto exp_data = std::make_shared<TensorImpl>(exp);
        out.inputs = {self_data, exp_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, exp_data](const TensorImpl& grad) {
            // d/da a^b = b * a^(b-1)
            if (self_data->require_grad) {
                TensorImpl lhs_grad = grad * *exp_data * self_data->pow(*exp_data - 1.0f);
                self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
            }
            // d/db a^b = a^b * log(a)
            if (exp_data->require_grad) {
                TensorImpl rhs_grad = grad * self_data->pow(*exp_data) * self_data->log();
                exp_data->gradient = std::make_shared<TensorImpl>(exp_data->gradient ? *exp_data->gradient + rhs_grad : rhs_grad);
            }
        };
    }

    return out;
}

TensorImpl TensorImpl::clip(float min, float max) const {
    TensorImpl out(shape, require_grad);
    for (int i = 0; i < nelem(); i++) {
        (*out.storage)[i] = std::clamp((*storage)[offset + i], min, max);
    }

    if (require_grad) {
        auto self_data = std::make_shared<TensorImpl>(*this);
        out.inputs = {self_data};
        out.is_leaf = false;
        out.gradient_func = [self_data, min, max](const TensorImpl& grad) {
            // gradient is 1 where input was in [min, max], 0 where it was clipped
            TensorImpl mask(self_data->shape, false);
            for (int i = 0; i < self_data->nelem(); i++) {
                float val = (*self_data->storage)[self_data->offset + i];
                (*mask.storage)[i] = (val >= min && val <= max) ? 1.0f : 0.0f;
            }
            TensorImpl lhs_grad = grad * mask;
            self_data->gradient = std::make_shared<TensorImpl>(self_data->gradient ? *self_data->gradient + lhs_grad : lhs_grad);
        };
    }

    return out;
}

TensorImpl TensorImpl::operator==(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a == b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

TensorImpl TensorImpl::operator!=(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a != b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

TensorImpl TensorImpl::operator<(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a < b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

TensorImpl TensorImpl::operator<=(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a <= b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

TensorImpl TensorImpl::operator>(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a > b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}

TensorImpl TensorImpl::operator>=(const TensorImpl& rhs) const {
    TensorImpl out = binary_op(rhs, [](const float a, const float b) { return a >= b ? 1.0f : 0.0f; });
    out.detach_grad_state();
    return out;
}


// TODO: Matt
TensorImpl& TensorImpl::grad() {
    return *gradient;
}

const TensorImpl& TensorImpl::grad() const {
    return *gradient;
}

bool TensorImpl::has_grad() const {
    return false;
}

void TensorImpl::zero_grad() {

}

void TensorImpl::backward() {
    if (require_grad) {
        gradient = std::make_shared<TensorImpl>(TensorImpl::ones(shape));
    } else {
        return;
    }

    // If there are no inputs, there can be no backpropagation
    if (inputs.size() == 0) {
        return;
    }

    // Build dependency count
    std::vector<std::shared_ptr<TensorImpl>> computation_tensors = inputs;

    for (int i = 0; i < computation_tensors.size(); ++i) {
        TensorImpl& tensor = *computation_tensors[i];

        // Increment dependency counter for all inputs
        for (const auto& input_tensor : tensor.inputs) {
            input_tensor->backprop_dep_count += 1;
            if (std::find(computation_tensors.begin(), computation_tensors.end(), input_tensor) == computation_tensors.end()) {
                // This tensor hasn't been seen before, so it needs to be tracked
                computation_tensors.push_back(input_tensor);
            }
        }
    }

    // Calculate gradients
    gradient_func_ptr(gradient);
    while (!computation_tensors.empty()) {
        auto tensor = computation_tensors.front();
        computation_tensors.erase(computation_tensors.begin());

        // If still waiting for dependencies, skip this
        if (tensor->backprop_dep_count != 0) {
            computation_tensors.push_back(tensor);
            continue;
        }

        // If the tensor is a leaf, there are no more upstream gradients to compute
        if (tensor->is_leaf) {
            continue;
        }

        // Calculate inputs' gradients and decrement dependency counts
        tensor->gradient_func_ptr(tensor->gradient);
        for (auto input : tensor->inputs) {
            input->backprop_dep_count -= 1;
        }
    }
}

void TensorImpl::set_gradient_func(GradientFunc func, const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
    this->gradient_func = func;
    this->inputs = inputs;
}

bool TensorImpl::get_is_leaf() const { return is_leaf; }

int TensorImpl::flat_index(const std::vector<int>& idx) const {
    if (idx.size() != shape.size()) {
        throw std::invalid_argument("Index dimensionality does not match tensor shape");
    }

    int flat = offset;

    for (std::size_t i = 0; i < idx.size(); ++i) {
        const int index = idx[i];

        if (index < 0 || index >= shape[i]) {
            throw std::out_of_range("TensorImpl index out of bounds");
        }
        flat += index * strides[i];
    }

    return flat;
}

void TensorImpl::detach_grad_state() {
    require_grad = false;
    is_leaf = true;
    inputs.clear();
    gradient_func = nullptr;
    gradient = nullptr;
}

void TensorImpl::compute_strides() {
    strides.resize(shape.size());

    int stride = 1;

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

int TensorImpl::broadcast_index(int flat, const std::vector<int>& out_shape, const std::vector<int>& in_shape) const {
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

std::vector<int> TensorImpl::broadcast_shape(const std::vector<int>& shape_one, const std::vector<int>& shape_two) {
    int ndim = std::max(shape_one.size(), shape_two.size());
    std::vector<int> broadcasted_shape(ndim);

    for (int i = 0; i < ndim; i++) {
        int shape_one_dim = i < shape_one.size() ? shape_one[shape_one.size() - 1 - i] : 1;
        int shape_two_dim = i < shape_two.size() ? shape_two[shape_two.size() - 1 - i] : 1;

        if (shape_one_dim == shape_two_dim || shape_one_dim == 1 || shape_two_dim == 1) {
            broadcasted_shape[ndim - 1 - i] = std::max(shape_one_dim, shape_two_dim);
        } else {
            throw std::invalid_argument("Incompatible dimensions for broadcasting: " + std::to_string(shape_one_dim) +
                                        " and " + std::to_string(shape_two_dim));
        }
    }

    return broadcasted_shape;
}
