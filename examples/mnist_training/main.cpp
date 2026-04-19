#include "core/TensorImpl.h"
#include "data/CSVLoader.h"
#include "loss/CrossEntropyLoss.h"
#include "nn/Linear.h"
#include "nn/activations/ReLU.h"
#include "nn/Serialization.h"
#include "optimizers/Adam.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int NUM_CLASSES = 10;

std::pair<Tensor, Tensor> load_and_normalize(const std::string &path) {
    auto [X_raw, y_raw] = axon::data::load_csv(path, 0);
    int rows = X_raw->size(0);
    int cols = X_raw->size(1);

    std::vector<float> x_data(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int c = 0; c < cols; ++c) {
            x_data[i * cols + c] = X_raw->at({i, c}) / 255.0f;
        }
    }

    std::vector<float> y_data(rows);
    for (int i = 0; i < rows; ++i) {
        y_data[i] = y_raw->at({i, 0});
    }

    return {TensorImpl::from_data(x_data, {rows, cols}, true),
            TensorImpl::from_data(y_data, {rows})};
}

std::pair<Tensor, Tensor> gather_batch(const std::vector<int> &indices, const Tensor &X, const Tensor &y) {
    int n = indices.size();
    int cols = X->size(1);
    std::vector<float> f_data(n * cols);
    std::vector<float> l_data(n);
    for (int i = 0; i < n; ++i) {
        int row = indices[i];
        for (int c = 0; c < cols; ++c) {
            f_data[i * cols + c] = X->at({row, c});
        }
        l_data[i] = y->at({row});
    }
    return {TensorImpl::from_data(f_data, {n, cols}, true),
            TensorImpl::from_data(l_data, {n})};
}

float calculate_accuracy(Linear &layer1, ReLU &relu, Linear &layer2, const Tensor &X, const Tensor &y) {
    Tensor h1 = layer1.forward(X);
    Tensor a1 = relu.forward(h1);
    Tensor logits = layer2.forward(a1);

    int correct = 0;
    int n = X->size(0);
    for (int i = 0; i < n; ++i) {
        float max_logit = -1e9f;
        int best_class = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float val = logits->at({i, c});
            if (val > max_logit) {
                max_logit = val;
                best_class = c;
            }
        }
        if (best_class == static_cast<int>(y->at({i}))) {
            correct++;
        }
    }
    return correct * 100.0f / n;
}

} // namespace

int main(int argc, char **argv) {
    std::string train_path = "mnist_train.csv";
    std::string test_path = "mnist_test.csv";
    if (argc >= 2) {
        train_path = argv[1];
    }
    if (argc >= 3) {
        test_path = argv[2];
    }

    auto [X_train, y_train] = load_and_normalize(train_path);
    auto [X_test, y_test] = load_and_normalize(test_path);

    int cols = X_train->size(1);
    if (X_test->size(1) != cols) {
        throw std::runtime_error("Train and test feature dimensions do not match");
    }
    int hidden_layer_size = 128;

    Linear layer1(cols, hidden_layer_size);
    ReLU relu;
    Linear layer2(hidden_layer_size, NUM_CLASSES);

    std::vector<Tensor> params;
    auto l1_p = layer1.parameters();
    auto l2_p = layer2.parameters();
    params.insert(params.end(), l1_p.begin(), l1_p.end());
    params.insert(params.end(), l2_p.begin(), l2_p.end());

    Adam optimizer(params, 0.001f);
    CrossEntropyLoss criterion;

    int batch_size = 64;
    int num_epochs = 10;
    int train_n = X_train->size(0);

    std::vector<int> perm(train_n);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(0);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::shuffle(perm.begin(), perm.end(), rng);
        float epoch_loss = 0.0f;
        int batches = 0;

        for (int start = 0; start + batch_size <= train_n; start += batch_size) {
            std::vector<int> batch_idx(perm.begin() + start, perm.begin() + start + batch_size);
            auto [Xb, yb] = gather_batch(batch_idx, X_train, y_train);

            Tensor h1 = layer1.forward(Xb);
            Tensor a1 = relu.forward(h1);
            Tensor logits = layer2.forward(a1);

            Tensor loss = criterion.forward(logits, yb);

            layer1.zero_grad();
            layer2.zero_grad();

            loss->backward();
            optimizer.step();

            epoch_loss += loss->at(0);
            ++batches;
        }

        std::cout << "Epoch " << (epoch + 1) << " Loss: " << (epoch_loss / batches) << std::endl;
    }

    float train_acc = calculate_accuracy(layer1, relu, layer2, X_train, y_train);
    std::cout << "Train accuracy: " << train_acc << "%" << std::endl;

    float test_acc = calculate_accuracy(layer1, relu, layer2, X_test, y_test);
    std::cout << "Test accuracy: " << test_acc << "%" << std::endl;

    std::string weights_path = "mnist_model.bin";
    axon::save(params, weights_path);
    std::cout << "Saved model to " << weights_path << std::endl;

    Linear layer1_loaded(cols, hidden_layer_size);
    Linear layer2_loaded(hidden_layer_size, NUM_CLASSES);

    std::vector<Tensor> loaded_params;
    auto l1_lp = layer1_loaded.parameters();
    auto l2_lp = layer2_loaded.parameters();
    loaded_params.insert(loaded_params.end(), l1_lp.begin(), l1_lp.end());
    loaded_params.insert(loaded_params.end(), l2_lp.begin(), l2_lp.end());

    axon::load(loaded_params, weights_path);
    std::cout << "Loaded model from " << weights_path << std::endl;

    ReLU relu_loaded;
    float loaded_test_acc = calculate_accuracy(layer1_loaded, relu_loaded, layer2_loaded, X_test, y_test);
    std::cout << "Loaded model test accuracy: " << loaded_test_acc << "%" << std::endl;

    return 0;
}