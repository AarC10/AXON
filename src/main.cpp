// NOTE THIS IS A TEMPORARY TEST FILE TO EXERCISE TENSORS AND SHOULD BE DELETED LATER
// Maybe one day we will have a G test
#include "core/TensorImpl.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
namespace {

bool approx_equal(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

void test_shape_and_fill() {
    Tensor zero_matrix = TensorImpl::zeros({2, 3});
    assert(zero_matrix->ndim() == 2);
    assert(zero_matrix->nelem() == 6);
    assert(zero_matrix->size(0) == 2);
    assert(zero_matrix->size(1) == 3);
    assert(zero_matrix->get_shape()[0] == 2);
    assert(zero_matrix->get_shape()[1] == 3);

    for (int i = 0; i < zero_matrix->nelem(); ++i) {
        assert(approx_equal(zero_matrix->at(i), 0.0f));
    }

    Tensor ones_matrix = TensorImpl::ones({4});
    for (int i = 0; i < ones_matrix->nelem(); ++i) {
        assert(approx_equal(ones_matrix->at(i), 1.0f));
    }

    Tensor full_matrix = TensorImpl::full({3}, 2.5f);
    for (int i = 0; i < full_matrix->nelem(); ++i) {
        assert(approx_equal(full_matrix->at(i), 2.5f));
    }
}

void test_requires_grad_flag() {
    Tensor tensor = TensorImpl::zeros({2, 2});
    assert(!tensor->get_require_grad());

    tensor->set_require_grad(true);
    assert(tensor->get_require_grad());

    tensor->set_require_grad(false);
    assert(!tensor->get_require_grad());
}

void test_inplace_arithmetic() {
    Tensor a = TensorImpl::full({3}, 2.0f);
    Tensor b = TensorImpl::full({3}, 3.0f);

    a += b; // 5
    for (int i = 0; i < a->nelem(); ++i) {
        assert(approx_equal(a->at(i), 5.0f));
    }

    a -= TensorImpl::full({3}, 1.5f); // 3.5
    for (int i = 0; i < a->nelem(); ++i) {
        assert(approx_equal(a->at(i), 3.5f));
    }

    a *= TensorImpl::full({3}, 2.0f); // 7
    for (int i = 0; i < a->nelem(); ++i) {
        assert(approx_equal(a->at(i), 7.0f));
    }

    a /= TensorImpl::full({3}, 2.0f); // 3.5
    for (int i = 0; i < a->nelem(); ++i) {
        assert(approx_equal(a->at(i), 3.5f));
    }
}

void test_copy_and_move_semantics() {
    Tensor src = TensorImpl::full({2}, 9.0f);
    Tensor copied = TensorImpl::copy(src);
    assert(approx_equal(copied->at(0), 9.0f));
    assert(approx_equal(copied->at(1), 9.0f));

    copied->at(0) = 4.0f;
    assert(approx_equal(src->at(0), 9.0f));
    assert(approx_equal(copied->at(0), 4.0f));

    Tensor moved(std::move(copied));
    assert(approx_equal(moved->at(0), 4.0f));
    assert(approx_equal(moved->at(1), 9.0f));
}

void test_backprop_basic() {
    auto a = TensorImpl::full({1}, 100.0f, true);
    auto b = TensorImpl::full({1},  10.0f, true);
    auto c = TensorImpl::full({1},   1.0f, true);
    auto d = a + b;
    auto e = b + c;
    auto f = d + e;
    auto g = d + f;

    g->backward();

    assert(approx_equal(a->grad()->at(0), 2.0f));
    assert(approx_equal(b->grad()->at(0), 3.0f));
    assert(approx_equal(c->grad()->at(0), 1.0f));
    assert(approx_equal(d->grad()->at(0), 2.0f));
    assert(approx_equal(e->grad()->at(0), 1.0f));
    assert(approx_equal(f->grad()->at(0), 1.0f));
    assert(approx_equal(g->grad()->at(0), 1.0f));
}

void test_backprop_arithmetic() {
    // Addition & Subtraction & Multiplication
    {
        auto a = TensorImpl::full({1}, 5.0f, true);
        auto b = TensorImpl::full({1}, 2.0f, true);
        auto c = (a * b) + (a - b); // 5*2 + (5-2) = 10 + 3 = 13
        // dc/da = b + 1 = 3
        // dc/db = a - 1 = 4
        c->backward();
        assert(approx_equal(a->grad()->at(0), 3.0f));
        assert(approx_equal(b->grad()->at(0), 4.0f));
    }
    
    // Division
    {
        auto a = TensorImpl::full({1}, 10.0f, true);
        auto b = TensorImpl::full({1}, 2.0f, true);
        auto c = a / b; // 5
        // dc/da = 1/b = 0.5
        // dc/db = -a/b^2 = -10/4 = -2.5
        c->backward();
        assert(approx_equal(a->grad()->at(0), 0.5f));
        assert(approx_equal(b->grad()->at(0), -2.5f));
    }
}

void test_backprop_math() {
    // Power (scalar)
    {
        auto a = TensorImpl::full({1}, 3.0f, true);
        auto b = pow(a, 2.0f); // 9
        // db/da = 2*a = 6
        b->backward();
        assert(approx_equal(a->grad()->at(0), 6.0f));
    }

    // Power (tensor)
    {
        auto a = TensorImpl::full({1}, 2.0f, true);
        auto b = TensorImpl::full({1}, 3.0f, true);
        auto c = pow(a, b); // 8
        // dc/da = b * a^(b-1) = 3 * 2^2 = 12
        // dc/db = a^b * log(a) = 2^3 * log(2) = 8 * log(2)
        c->backward();
        assert(approx_equal(a->grad()->at(0), 12.0f));
        assert(approx_equal(b->grad()->at(0), 8.0f * std::log(2.0f)));
    }

    // Exp & Log
    {
        auto a = TensorImpl::full({1}, 2.0f, true);
        auto b = exp(a); // e^2
        // db/da = e^2
        b->backward();
        assert(approx_equal(a->grad()->at(0), std::exp(2.0f)));

        auto x = TensorImpl::full({1}, 2.0f, true);
        auto y = log(x); // log(2)
        // dy/dx = 1/x = 0.5
        y->backward();
        assert(approx_equal(x->grad()->at(0), 0.5f));
    }

    // Sqrt
    {
        auto a = TensorImpl::full({1}, 4.0f, true);
        auto b = sqrt(a); // 2
        // db/da = 1 / (2 * sqrt(a)) = 1 / (2 * 2) = 0.25
        b->backward();
        assert(approx_equal(a->grad()->at(0), 0.25f));
    }
}

void test_backprop_complex() {
    // a = 2, b = 3
    // x = a * b = 6
    // y = exp(x) = exp(6)
    // z = y + a = exp(6) + 2
    // dz/da = dz/dy * dy/dx * dx/da + dz/da_direct
    //       = 1 * exp(6) * b + 1 = 3 * exp(6) + 1
    // dz/db = dz/dy * dy/dx * dx/db
    //       = 1 * exp(6) * a = 2 * exp(6)
    
    auto a = TensorImpl::full({1}, 2.0f, true);
    auto b = TensorImpl::full({1}, 3.0f, true);
    auto x = a * b;
    auto y = exp(x);
    auto z = y + a;
    
    z->backward();
    
    assert(approx_equal(a->grad()->at(0), 3.0f * std::exp(6.0f) + 1.0f));
    assert(approx_equal(b->grad()->at(0), 2.0f * std::exp(6.0f)));
}

void test_backprop_clip_abs() {
    // Abs
    {
        auto a = TensorImpl::full({1}, -5.0f, true);
        auto b = abs(a); // 5
        // db/da = -1
        b->backward();
        assert(approx_equal(a->grad()->at(0), -1.0f));

        auto c = TensorImpl::full({1}, 5.0f, true);
        auto d = abs(c); // 5
        // dd/dc = 1
        d->backward();
        assert(approx_equal(c->grad()->at(0), 1.0f));
    }

    // Clip
    {
        auto a = TensorImpl::full({1}, 10.0f, true);
        auto b = clip(a, 0.0f, 5.0f); // 5
        // db/da = 0 (since it's clipped)
        b->backward();
        assert(approx_equal(a->grad()->at(0), 0.0f));

        auto c = TensorImpl::full({1}, 3.0f, true);
        auto d = clip(c, 0.0f, 5.0f); // 3
        // dd/dc = 1 (since it's not clipped)
        d->backward();
        assert(approx_equal(c->grad()->at(0), 1.0f));
    }
}

} // namespace

int main() {
    test_shape_and_fill();
    test_requires_grad_flag();
    test_inplace_arithmetic();
    test_copy_and_move_semantics();
    test_backprop_basic();
    test_backprop_arithmetic();
    test_backprop_math();
    test_backprop_complex();
    test_backprop_clip_abs();

    std::cout << "Pass!\n";

    return 0;
}
