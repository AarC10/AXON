// NOTE THIS IS A TEMPORARY TEST FILE TO EXERCISE TENSORS AND SHOULD BE DELETED LATER
// Maybe one day we will have a G test
#include "core/Tensor.h"
#include "loss/CrossEntropyLoss.h"
#include "loss/MSELoss.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
namespace {

bool approx_equal(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

void test_shape_and_fill() {
    Tensor zero_matrix = Tensor::zeros({2, 3});
    assert(zero_matrix.ndim() == 2);
    assert(zero_matrix.nelem() == 6);
    assert(zero_matrix.size(0) == 2);
    assert(zero_matrix.size(1) == 3);
    assert(zero_matrix.get_shape()[0] == 2);
    assert(zero_matrix.get_shape()[1] == 3);

    for (int i = 0; i < zero_matrix.nelem(); ++i) {
        assert(approx_equal(zero_matrix[i], 0.0f));
    }

    Tensor ones_matrix = Tensor::ones({4});
    for (int i = 0; i < ones_matrix.nelem(); ++i) {
        assert(approx_equal(ones_matrix[i], 1.0f));
    }

    Tensor full_matrix = Tensor::full({3}, 2.5f);
    for (int i = 0; i < full_matrix.nelem(); ++i) {
        assert(approx_equal(full_matrix[i], 2.5f));
    }
}

void test_requires_grad_flag() {
    Tensor tensor = Tensor::zeros({2, 2});
    assert(!tensor.get_require_grad());

    tensor.set_require_grad(true);
    assert(tensor.get_require_grad());

    tensor.set_require_grad(false);
    assert(!tensor.get_require_grad());
}

void test_inplace_arithmetic() {
    Tensor a = Tensor::full({3}, 2.0f);
    Tensor b = Tensor::full({3}, 3.0f);

    a += b; // 5
    for (int i = 0; i < a.nelem(); ++i) {
        assert(approx_equal(a[i], 5.0f));
    }

    a -= Tensor::full({3}, 1.5f); // 3.5
    for (int i = 0; i < a.nelem(); ++i) {
        assert(approx_equal(a[i], 3.5f));
    }

    a *= Tensor::full({3}, 2.0f); // 7
    for (int i = 0; i < a.nelem(); ++i) {
        assert(approx_equal(a[i], 7.0f));
    }

    a /= Tensor::full({3}, 2.0f); // 3.5
    for (int i = 0; i < a.nelem(); ++i) {
        assert(approx_equal(a[i], 3.5f));
    }
}

void test_copy_and_move_semantics() {
    Tensor src = Tensor::full({2}, 9.0f);
    Tensor copied(src);
    assert(approx_equal(copied[0], 9.0f));
    assert(approx_equal(copied[1], 9.0f));

    copied[0] = 4.0f;
    assert(approx_equal(src[0], 9.0f));
    assert(approx_equal(copied[0], 4.0f));

    Tensor moved(std::move(copied));
    assert(approx_equal(moved[0], 4.0f));
    assert(approx_equal(moved[1], 9.0f));
}

void test_mse_loss() {
    // Mean of squared differences: ((1-0)^2 + (2-0)^2 + (3-0)^2 + (4-0)^2) / 4 = 30 / 4 = 7.5
    Tensor prediction({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor target = Tensor::zeros({2, 2});
    MSELoss criterion;
    Tensor loss = criterion.forward(prediction, target);
    assert(loss.nelem() == 1);
    assert(approx_equal(loss[0], 7.5f));

    // Identical tensors should produce a zero loss
    Tensor identicalLoss = criterion.forward(prediction, prediction);
    assert(approx_equal(identicalLoss[0], 0.0f));
}

void test_cross_entropy_loss() {
    // Two samples with three classes; correct class gets a much larger logit
    Tensor logits({2.0f, 1.0f, 0.1f, 0.1f, 1.0f, 2.0f}, {2, 3});
    Tensor targets(std::vector<float>{0.0f, 2.0f}, std::vector<int>{2});
    CrossEntropyLoss criterion;
    Tensor loss = criterion.forward(logits, targets);
    assert(loss.nelem() == 1);

    // Hand-computed reference using log-sum-exp on each row
    const float row0LogSumExp = std::log(std::exp(2.0f) + std::exp(1.0f) + std::exp(0.1f));
    const float row1LogSumExp = std::log(std::exp(0.1f) + std::exp(1.0f) + std::exp(2.0f));
    const float expectedLoss = ((row0LogSumExp - 2.0f) + (row1LogSumExp - 2.0f)) / 2.0f;
    assert(approx_equal(loss[0], expectedLoss, 1e-4f));

    // Confidently correct predictions should give a near-zero loss
    Tensor confidentLogits({10.0f, -10.0f, -10.0f, -10.0f, 10.0f, -10.0f}, {2, 3});
    Tensor confidentTargets(std::vector<float>{0.0f, 1.0f}, std::vector<int>{2});
    Tensor confidentLoss = criterion.forward(confidentLogits, confidentTargets);
    assert(confidentLoss[0] < 1e-3f);
}

} // namespace

int main() {
    test_shape_and_fill();
    test_requires_grad_flag();
    test_inplace_arithmetic();
    test_copy_and_move_semantics();
    test_mse_loss();
    test_cross_entropy_loss();

    std::cout << "Pass!\n";

    return 0;
}
