// NOTE THIS IS A TEMPORARY TEST FILE TO EXERCISE TENSORS AND SHOULD BE DELETED LATER
// Maybe one day we will have a G test
#include "core/TensorImpl.h"
#include "data/CSVLoader.h"
#include "loss/CrossEntropyLoss.h"
#include "loss/MSELoss.h"
#include "nn/activations/ReLU.h"
#include "nn/activations/Sigmoid.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

bool approx_equal(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

void print_tensor(const Tensor &tensor, const std::string &name) {
    std::cout << name << " shape: {";

    const std::vector<int> &shape = tensor->get_shape();

    for (std::size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];

        if (i + 1 < shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "}\n";

    if (shape.size() != 2) {
        for (int i = 0; i < tensor->nelem(); ++i) {
            std::cout << tensor->at(i) << "\n";
        }

        return;
    }

    int rows = shape[0];
    int cols = shape[1];

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            std::cout << tensor->at(index);

            if (col + 1 < cols) {
                std::cout << ", ";
            }
        }

        std::cout << "\n";
    }
}

double run_benchmark(std::string_view name, const std::function<void()> &test) {
    const auto start = Clock::now();
    test();
    const auto end = Clock::now();

    const auto elapsed = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << std::left << std::setw(32) << name << " " << std::fixed << std::setprecision(2) << elapsed << " us\n";
    return elapsed;
}

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
    auto b = TensorImpl::full({1}, 10.0f, true);
    auto c = TensorImpl::full({1}, 1.0f, true);
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

void test_matmul() {
    {
        Tensor lhs = TensorImpl::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
        Tensor rhs = TensorImpl::from_data({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
        Tensor out = matmul(lhs, rhs);

        assert(out->get_shape() == std::vector<int>({2, 2}));
        assert(approx_equal(out->at(0), 19.0f));
        assert(approx_equal(out->at(1), 22.0f));
        assert(approx_equal(out->at(2), 43.0f));
        assert(approx_equal(out->at(3), 50.0f));
    }

    {
        Tensor lhs = TensorImpl::from_data({1.0f, 2.0f}, {2}, true);
        Tensor rhs = TensorImpl::from_data({3.0f, 4.0f}, {2}, true);
        Tensor out = matmul(lhs, rhs);

        assert(out->get_shape() == std::vector<int>({1}));
        assert(approx_equal(out->at(0), 11.0f));

        out->backward();
        assert(approx_equal(lhs->grad()->at(0), 3.0f));
        assert(approx_equal(lhs->grad()->at(1), 4.0f));
        assert(approx_equal(rhs->grad()->at(0), 1.0f));
        assert(approx_equal(rhs->grad()->at(1), 2.0f));
    }
}

void test_relu() {
    ReLU relu;
    Tensor x = TensorImpl::from_data({-1.0f, 0.0f, 2.0f}, {3}, true);
    Tensor y = relu.forward(*x);

    assert(approx_equal(y->at(0), 0.0f));
    assert(approx_equal(y->at(1), 0.0f));
    assert(approx_equal(y->at(2), 2.0f));

    y->backward();
    assert(approx_equal(x->grad()->at(0), 0.0f));
    assert(approx_equal(x->grad()->at(1), 0.0f));
    assert(approx_equal(x->grad()->at(2), 1.0f));
}

void test_sigmoid() {
    Sigmoid sigmoid;
    Tensor x = TensorImpl::from_data({0.0f}, {1}, true);
    Tensor y = sigmoid.forward(*x);

    assert(approx_equal(y->at(0), 0.5f));

    y->backward();
    assert(approx_equal(x->grad()->at(0), 0.25f));
}

void test_mse_loss() {
    // Mean of squared differences: ((1-0)^2 + (2-0)^2 + (3-0)^2 + (4-0)^2) / 4 = 30 / 4 = 7.5
    Tensor prediction = TensorImpl::from_data({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor target = TensorImpl::zeros({2, 2});
    MSELoss criterion;
    Tensor loss = criterion.forward(prediction, target);
    assert(loss->nelem() == 1);
    assert(approx_equal(loss->at(0), 7.5f));

    // Identical tensors should produce a zero loss
    Tensor identicalLoss = criterion.forward(prediction, prediction);
    assert(approx_equal(identicalLoss->at(0), 0.0f));
}

void test_cross_entropy_loss() {
    // Two samples with three classes, correct class gets a much larger logit
    Tensor logits = TensorImpl::from_data({2.0f, 1.0f, 0.1f, 0.1f, 1.0f, 2.0f}, {2, 3});
    Tensor targets = TensorImpl::from_data(std::vector<float>{0.0f, 2.0f}, std::vector<int>{2});
    CrossEntropyLoss criterion;
    Tensor loss = criterion.forward(logits, targets);
    assert(loss->nelem() == 1);

    // Hand-computed reference using log-sum-exp on each row
    const float row0LogSumExp = std::log(std::exp(2.0f) + std::exp(1.0f) + std::exp(0.1f));
    const float row1LogSumExp = std::log(std::exp(0.1f) + std::exp(1.0f) + std::exp(2.0f));
    const float expectedLoss = ((row0LogSumExp - 2.0f) + (row1LogSumExp - 2.0f)) / 2.0f;
    assert(approx_equal(loss->at(0), expectedLoss, 1e-4f));

    // Confidently correct predictions should give a near-zero loss
    Tensor confidentLogits = TensorImpl::from_data({10.0f, -10.0f, -10.0f, -10.0f, 10.0f, -10.0f}, {2, 3});
    Tensor confidentTargets = TensorImpl::from_data(std::vector<float>{0.0f, 1.0f}, std::vector<int>{2});
    Tensor confidentLoss = criterion.forward(confidentLogits, confidentTargets);
    assert(confidentLoss->at(0) < 1e-3f);
}

} // namespace

int main(int argc, char **argv) {
    double total_us = 0.0;

    std::cout << "Benchmark results:\n";
    total_us += run_benchmark("test_shape_and_fill", test_shape_and_fill);
    total_us += run_benchmark("test_requires_grad_flag", test_requires_grad_flag);
    total_us += run_benchmark("test_inplace_arithmetic", test_inplace_arithmetic);
    total_us += run_benchmark("test_copy_and_move_semantics", test_copy_and_move_semantics);
    total_us += run_benchmark("test_backprop_basic", test_backprop_basic);
    total_us += run_benchmark("test_backprop_arithmetic", test_backprop_arithmetic);
    total_us += run_benchmark("test_backprop_math", test_backprop_math);
    total_us += run_benchmark("test_backprop_complex", test_backprop_complex);
    total_us += run_benchmark("test_backprop_clip_abs", test_backprop_clip_abs);
    total_us += run_benchmark("test_matmul", test_matmul);
    total_us += run_benchmark("test_relu", test_relu);
    total_us += run_benchmark("test_relu", test_sigmoid);
    total_us += run_benchmark("test_mse_loss", test_mse_loss);
    total_us += run_benchmark("test_cross_entropy_loss", test_cross_entropy_loss);

    std::cout << std::left << std::setw(32) << "total" << " " << std::fixed << std::setprecision(2) << total_us
              << " us\n";
    std::cout << "Pass!\n";

    if (argc >= 3) {
        std::string path = argv[1];
        int label_col = std::stoi(argv[2]);
        std::pair<Tensor, Tensor> loaded = axon::data::load_csv(path, label_col);

        print_tensor(loaded.first, "X");
        print_tensor(loaded.second, "y");
    }

    return 0;
}
