#include "core/TensorImpl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

Tensor TensorImpl::from_data(const std::vector<float>& data, const std::vector<int>& shape, bool require_grad) {
    return Tensor(new TensorImpl(data, shape, require_grad));
}

Tensor TensorImpl::zeros(const std::vector<int>& shape, bool require_grad) {
    // Zero inited in ctor
    return Tensor(new TensorImpl(shape, require_grad));
}

Tensor TensorImpl::ones(const std::vector<int>& shape, bool require_grad) {
    return TensorImpl::full(shape, 1.0f, require_grad);
}

Tensor TensorImpl::full(const std::vector<int>& shape, float value, bool requires_grad) {
    Tensor tensor = Tensor(new TensorImpl(shape, requires_grad));
    std::fill(tensor->storage->begin(), tensor->storage->end(), value);
    return tensor;
}

Tensor TensorImpl::randn(const std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    Tensor tensor = Tensor(new TensorImpl(shape, require_grad));

    for (float& x : *tensor->storage) {
        x = dis(gen);
    }

    return tensor;
}

Tensor TensorImpl::rand(const std::vector<int>& shape, bool require_grad) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    Tensor tensor = Tensor(new TensorImpl(shape, require_grad));

    for (float& x : *tensor->storage) {
        x = dis(gen);
    }

    return tensor;
}

Tensor TensorImpl::eye(int n, bool require_grad) {
    Tensor tensor = Tensor(new TensorImpl(std::vector<int>{n, n}, require_grad));
    for (int i = 0; i < n; ++i) {
        (*tensor->storage)[i * n + i] = 1.0f;
    }
    return tensor;
}

Tensor TensorImpl::arange(float start, float stop, float step, bool require_grad) {
    if (step == 0.0f) {
        throw std::invalid_argument("Step must be non-zero");
    }

    int n = static_cast<int>(std::ceil((stop - start) / step));

    if (n <= 0) {
        throw std::invalid_argument("Invalid range: start=" + std::to_string(start) + ", stop=" + std::to_string(stop) +
                                    "), step=" + std::to_string(step));
    }

    Tensor tensor = Tensor(new TensorImpl(std::vector<int>({n}), require_grad));
    for (int i = 0; i < n; ++i) {
        (*tensor->storage)[i] = start + i * step;
    }

    return tensor;
}

Tensor TensorImpl::copy(const Tensor& other) { return Tensor(new TensorImpl(*other)); }

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

float TensorImpl::at(int idx) const { return (*storage)[offset + idx]; }

float& TensorImpl::at(int idx) { return (*storage)[offset + idx]; }

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a + b; });

    if (out->require_grad) {
        out->inputs = {lhs, rhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, rhs](const Tensor& grad) {
            // Adding should pass grad straight through both sides
            if (lhs->require_grad) {
                lhs->gradient = lhs->gradient ? lhs->gradient + grad : grad;
            }

            if (rhs->require_grad) {
                rhs->gradient = rhs->gradient ? rhs->gradient + grad : grad;
            }
        };
    }

    return out;
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a - b; });

    if (out->require_grad) {
        out->inputs = {lhs, rhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, rhs](const Tensor& grad) {
            // lhs should get +grad
            if (lhs->require_grad) {
                lhs->gradient = lhs->gradient ? lhs->gradient + grad : grad;
            }

            // rhs gets -grad
            if (rhs->require_grad) {
                Tensor neg_grad = -grad;
                rhs->gradient = rhs->gradient ? rhs->gradient + neg_grad : neg_grad;
            }
        };
    }

    return out;
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a * b; });

    if (out->require_grad) {
        out->inputs = {lhs, rhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, rhs](const Tensor& grad) {
            // d/da (a*b) = b
            // d/db (a*b) = a
            if (lhs->require_grad) {
                Tensor lhs_grad = grad * rhs;
                lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
            }
            if (rhs->require_grad) {
                Tensor rhs_grad = grad * lhs;
                rhs->gradient = rhs->gradient ? rhs->gradient + rhs_grad : rhs_grad;
            }
        };
    }

    return out;
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a / b; });

    if (out->require_grad) {
        out->inputs = {lhs, rhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, rhs](const Tensor& grad) {
            // d/da (a/b) = 1/b
            // d/db (a/b) = -a/b^2
            if (lhs->require_grad) {
                Tensor lhs_grad = grad / rhs;
                lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
            }
            if (rhs->require_grad) {
                Tensor rhs_grad = grad * (lhs * -1.0f) / (rhs * rhs);
                rhs->gradient = rhs->gradient ? rhs->gradient + rhs_grad : rhs_grad;
            }
        };
    }

    return out;
}

Tensor operator-(const Tensor& tensor) { return tensor * -1.0f; }

Tensor operator+(const Tensor& tensor, float scalar) { return tensor + TensorImpl::full(tensor->shape, scalar); }

Tensor operator-(const Tensor& tensor, float scalar) { return tensor - TensorImpl::full(tensor->shape, scalar); }

Tensor operator*(const Tensor& tensor, float scalar) { return tensor * TensorImpl::full(tensor->shape, scalar); }

Tensor operator/(const Tensor& tensor, float scalar) { return tensor / TensorImpl::full(tensor->shape, scalar); }

Tensor& operator+=(Tensor& lhs, const ConstTensor& rhs) {
    if (lhs->require_grad && !lhs->is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (lhs->shape != rhs->shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < lhs->nelem(); ++i) {
        (*lhs->storage)[lhs->offset + i] += (*rhs->storage)[rhs->offset + i];
    }

    return lhs;
}

Tensor& operator-=(Tensor& lhs, const ConstTensor& rhs) {
    if (lhs->require_grad && !lhs->is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (lhs->shape != rhs->shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < lhs->nelem(); ++i) {
        (*lhs->storage)[lhs->offset + i] -= (*rhs->storage)[rhs->offset + i];
    }

    return lhs;
}

Tensor& operator*=(Tensor& lhs, const ConstTensor& rhs) {
    if (lhs->require_grad && !lhs->is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (lhs->shape != rhs->shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < lhs->nelem(); ++i) {
        (*lhs->storage)[lhs->offset + i] *= (*rhs->storage)[rhs->offset + i];
    }

    return lhs;
}

Tensor& operator/=(Tensor& lhs, const ConstTensor& rhs) {
    if (lhs->require_grad && !lhs->is_leaf) {
        throw std::runtime_error("In-place op on non-leaf tensor that requires grad will corrupt the autograd graph");
    }

    if (lhs->shape != rhs->shape) {
        throw std::invalid_argument("In-place op requires identical shapes");
    }

    for (int i = 0; i < lhs->nelem(); ++i) {
        (*lhs->storage)[lhs->offset + i] /= (*rhs->storage)[rhs->offset + i];
    }

    return lhs;
}

Tensor operator+(float scalar, const Tensor& tensor) { return tensor + scalar; }

Tensor operator-(float scalar, const Tensor& tensor) { return TensorImpl::full(tensor->shape, scalar) - tensor; }

Tensor operator*(float scalar, const Tensor& tensor) { return tensor * scalar; }

Tensor operator/(float scalar, const Tensor& tensor) { return TensorImpl::full(tensor->shape, scalar) / tensor; }

Tensor exp(const Tensor& lhs) {
    Tensor out = Tensor(new TensorImpl(lhs->shape, lhs->require_grad));

    for (int i = 0; i < lhs->nelem(); ++i) {
        (*out->storage)[i] = std::exp((*lhs->storage)[lhs->offset + i]);
    }

    if (lhs->require_grad) {
        out->inputs = {lhs};
        out->is_leaf = false;
        out->gradient_func = [lhs](const Tensor& grad) {
            // d/dx exp(x) = exp(x)
            Tensor lhs_grad = grad * exp(lhs);
            lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
        };
    }

    return out;
}

Tensor log(const Tensor& lhs) {
    Tensor out = Tensor(new TensorImpl(lhs->shape, lhs->require_grad));

    for (int i = 0; i < lhs->nelem(); ++i) {
        (*out->storage)[i] = std::log((*lhs->storage)[lhs->offset + i]);
    }

    if (lhs->require_grad) {
        out->inputs = {lhs};
        out->is_leaf = false;
        out->gradient_func = [lhs](const Tensor& grad) {
            // d/dx log(x) = 1/x
            Tensor lhs_grad = grad / lhs;
            lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
        };
    }

    return out;
}

Tensor sqrt(const Tensor& lhs) {
    Tensor out = Tensor(new TensorImpl(lhs->shape, lhs->require_grad));
    for (int i = 0; i < lhs->nelem(); ++i) {
        (*out->storage)[i] = std::sqrt((*lhs->storage)[lhs->offset + i]);
    }

    if (lhs->require_grad) {
        out->inputs = {lhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, out](const Tensor& grad) {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x))
            Tensor lhs_grad = grad / (out * 2.0f);
            lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
        };
    }

    return out;
}
Tensor abs(const Tensor& lhs) {
    Tensor out = Tensor(new TensorImpl(lhs->shape, lhs->require_grad));
    for (int i = 0; i < lhs->nelem(); i++) {
        (*out->storage)[i] = std::abs((*lhs->storage)[lhs->offset + i]);
    }

    if (lhs->require_grad) {
        out->inputs = {lhs};
        out->is_leaf = false;
        out->gradient_func = [lhs](const Tensor& grad) {
            // d/dx abs(x) = sign(x)
            Tensor sign = Tensor(new TensorImpl(lhs->shape, false));
            for (int i = 0; i < lhs->nelem(); i++) {
                float val = (*lhs->storage)[lhs->offset + i];
                (*sign->storage)[i] = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
            }
            Tensor lhs_grad = grad * sign;
            lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
        };
    }

    return out;
}

Tensor pow(const Tensor& lhs, float exponent) {
    Tensor out = Tensor(new TensorImpl(lhs->shape, lhs->require_grad));
    for (int i = 0; i < lhs->nelem(); i++) {
        (*out->storage)[i] = std::pow((*lhs->storage)[lhs->offset + i], exponent);
    }

    if (lhs->require_grad) {
        out->inputs = {lhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, exponent](const Tensor& grad) {
            // d/dx x^n = n * x^(n-1)
            Tensor lhs_grad = grad * pow(lhs, exponent - 1.0f) * exponent;
            lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
        };
    }

    return out;
}

Tensor pow(const Tensor& lhs, const Tensor& exp) {
    Tensor out = lhs->binary_op(exp, [](const float a, const float b) { return std::pow(a, b); });

    if (out->require_grad) {
        out->inputs = {lhs, exp};
        out->is_leaf = false;
        out->gradient_func = [lhs, exp](const Tensor& grad) {
            // d/da a^b = b * a^(b-1)
            if (lhs->require_grad) {
                Tensor lhs_grad = grad * exp * pow(lhs, exp - 1.0f);
                lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
            }
            // d/db a^b = a^b * log(a)
            if (exp->require_grad) {
                Tensor rhs_grad = grad * pow(lhs, exp) * log(lhs);
                exp->gradient = exp->gradient ? exp->gradient + rhs_grad : rhs_grad;
            }
        };
    }

    return out;
}

Tensor clip(const Tensor& lhs, float min, float max) {
    Tensor out = Tensor(new TensorImpl(lhs->shape, lhs->require_grad));
    for (int i = 0; i < lhs->nelem(); i++) {
        (*out->storage)[i] = std::clamp((*lhs->storage)[lhs->offset + i], min, max);
    }

    if (lhs->require_grad) {
        out->inputs = {lhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, min, max](const Tensor& grad) {
            // gradient is 1 where input was in [min, max], 0 where it was clipped
            Tensor mask = Tensor(new TensorImpl(lhs->shape, false));
            for (int i = 0; i < lhs->nelem(); i++) {
                float val = (*lhs->storage)[lhs->offset + i];
                (*mask->storage)[i] = (val >= min && val <= max) ? 1.0f : 0.0f;
            }
            Tensor lhs_grad = grad * mask;
            lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
        };
    }

    return out;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    const int lhs_dims = lhs->ndim();
    const int rhs_dims = rhs->ndim();

    if ((lhs_dims != 1 && lhs_dims != 2) || (rhs_dims != 1 && rhs_dims != 2)) {
        throw std::invalid_argument("matmul currently supports only 1D or 2D tensors");
    }

    const int lhs_rows = lhs_dims == 1 ? 1 : lhs->size(0);
    const int lhs_cols = lhs_dims == 1 ? lhs->size(0) : lhs->size(1);
    const int rhs_rows = rhs->size(0);
    const int rhs_cols = rhs_dims == 1 ? 1 : rhs->size(1);

    if (lhs_cols != rhs_rows) {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    std::vector<int> out_shape;
    if (lhs_dims == 1 && rhs_dims == 1) {
        out_shape = {1};
    } else if (lhs_dims == 1 && rhs_dims == 2) {
        out_shape = {rhs_cols};
    } else if (lhs_dims == 2 && rhs_dims == 1) {
        out_shape = {lhs_rows};
    } else {
        out_shape = {lhs_rows, rhs_cols};
    }

    Tensor out = Tensor(new TensorImpl(out_shape, lhs->require_grad || rhs->require_grad));

    for (int row = 0; row < lhs_rows; ++row) {
        for (int col = 0; col < rhs_cols; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < lhs_cols; ++k) {
                const int lhs_index = lhs_dims == 1 ? k : row * lhs_cols + k;
                const int rhs_index = rhs_dims == 1 ? k : k * rhs_cols + col;
                sum += lhs->at(lhs_index) * rhs->at(rhs_index);
            }

            const int out_index = (lhs_dims == 1 && rhs_dims == 1)   ? 0
                                  : (lhs_dims == 1 && rhs_dims == 2) ? col
                                  : (lhs_dims == 2 && rhs_dims == 1) ? row
                                                                     : row * rhs_cols + col;
            out->at(out_index) = sum;
        }
    }

    if (out->require_grad) {
        out->inputs = {lhs, rhs};
        out->is_leaf = false;
        out->gradient_func = [lhs, rhs, lhs_dims, rhs_dims, lhs_rows, lhs_cols, rhs_cols](const Tensor& grad) {
            if (lhs->require_grad) {
                Tensor lhs_grad = TensorImpl::zeros(lhs->get_shape());

                for (int row = 0; row < lhs_rows; ++row) {
                    for (int k = 0; k < lhs_cols; ++k) {
                        float sum = 0.0f;
                        for (int col = 0; col < rhs_cols; ++col) {
                            const int grad_index = (lhs_dims == 1 && rhs_dims == 1)   ? 0
                                                   : (lhs_dims == 1 && rhs_dims == 2) ? col
                                                   : (lhs_dims == 2 && rhs_dims == 1) ? row
                                                                                      : row * rhs_cols + col;
                            const int rhs_index = rhs_dims == 1 ? k : k * rhs_cols + col;
                            sum += grad->at(grad_index) * rhs->at(rhs_index);
                        }

                        const int lhs_index = lhs_dims == 1 ? k : row * lhs_cols + k;
                        lhs_grad->at(lhs_index) = sum;
                    }
                }

                lhs->gradient = lhs->gradient ? lhs->gradient + lhs_grad : lhs_grad;
            }

            if (rhs->require_grad) {
                Tensor rhs_grad = TensorImpl::zeros(rhs->get_shape());
                const int rhs_rows_local = rhs->size(0);
                const int rhs_cols_local = rhs_dims == 1 ? 1 : rhs->size(1);

                for (int k = 0; k < rhs_rows_local; ++k) {
                    for (int col = 0; col < rhs_cols_local; ++col) {
                        float sum = 0.0f;
                        for (int row = 0; row < lhs_rows; ++row) {
                            const int lhs_index = lhs_dims == 1 ? k : row * lhs_cols + k;
                            const int grad_index = (lhs_dims == 1 && rhs_dims == 1)   ? 0
                                                   : (lhs_dims == 1 && rhs_dims == 2) ? col
                                                   : (lhs_dims == 2 && rhs_dims == 1) ? row
                                                                                      : row * rhs_cols + col;
                            sum += lhs->at(lhs_index) * grad->at(grad_index);
                        }

                        const int rhs_index = rhs_dims == 1 ? k : k * rhs_cols_local + col;
                        rhs_grad->at(rhs_index) = sum;
                    }
                }

                rhs->gradient = rhs->gradient ? rhs->gradient + rhs_grad : rhs_grad;
            }
        };
    }

    return out;
}

Tensor operator==(const ConstTensor& lhs, const ConstTensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a == b ? 1.0f : 0.0f; });
    out->detach_grad_state();
    return out;
}

Tensor operator!=(const ConstTensor& lhs, const ConstTensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a != b ? 1.0f : 0.0f; });
    out->detach_grad_state();
    return out;
}

Tensor operator<(const ConstTensor& lhs, const ConstTensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a < b ? 1.0f : 0.0f; });
    out->detach_grad_state();
    return out;
}

Tensor operator<=(const ConstTensor& lhs, const ConstTensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a <= b ? 1.0f : 0.0f; });
    out->detach_grad_state();
    return out;
}

Tensor operator>(const ConstTensor& lhs, const ConstTensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a > b ? 1.0f : 0.0f; });
    out->detach_grad_state();
    return out;
}

Tensor operator>=(const ConstTensor& lhs, const ConstTensor& rhs) {
    Tensor out = lhs->binary_op(rhs, [](const float a, const float b) { return a >= b ? 1.0f : 0.0f; });
    out->detach_grad_state();
    return out;
}

// TODO: Matt
Tensor& TensorImpl::grad() { return gradient; }

bool TensorImpl::has_grad() const { return gradient != nullptr; }

void TensorImpl::zero_grad() { gradient = nullptr; }

void TensorImpl::backward() {
    if (require_grad) {
        gradient = TensorImpl::ones(shape);
    } else {
        return;
    }

    // If there are no inputs, there can be no backpropagation
    if (inputs.size() == 0) {
        return;
    }

    // Build dependency count
    std::vector<Tensor> computation_tensors = inputs;

    for (int i = 0; i < computation_tensors.size(); ++i) {
        TensorImpl& tensor = *computation_tensors[i];

        // Increment dependency counter for all inputs
        for (const auto& input_tensor : tensor.inputs) {
            input_tensor->backprop_dep_count += 1;
            if (std::find(computation_tensors.begin(), computation_tensors.end(), input_tensor) ==
                computation_tensors.end()) {
                // This tensor hasn't been seen before, so it needs to be tracked
                computation_tensors.push_back(input_tensor);
            }
        }
    }

    // Calculate gradients
    gradient_func(gradient);
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
        tensor->gradient_func(tensor->gradient);
        for (auto input : tensor->inputs) {
            input->backprop_dep_count -= 1;
        }
    }
}

void TensorImpl::set_gradient_func(GradientFunc func, const std::vector<Tensor>& inputs) {
    this->gradient_func = func;
    this->inputs = inputs;
    is_leaf = inputs.empty();
}

bool TensorImpl::get_is_leaf() const { return is_leaf; }

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
      inputs(other.inputs), gradient_func(other.gradient_func) {}

TensorImpl::TensorImpl(TensorImpl&& other)
    : storage(std::move(other.storage)), offset(other.offset), shape(std::move(other.shape)),
      strides(std::move(other.strides)), require_grad(other.require_grad), is_leaf(other.is_leaf),
      gradient(std::move(other.gradient)), inputs(std::move(other.inputs)),
      gradient_func(std::move(other.gradient_func)) {}

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
