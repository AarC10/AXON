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

void test_backprop() {
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

} // namespace

int main() {
    //test_shape_and_fill();
    //test_requires_grad_flag();
    //test_inplace_arithmetic();
    test_copy_and_move_semantics();
    test_backprop();

    std::cout << "Pass!\n";

    return 0;
}
