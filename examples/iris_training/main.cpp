#include "core/TensorImpl.h"
#include "data/CSVLoader.h"
#include "loss/CrossEntropyLoss.h"
#include "nn/Linear.h"
#include "nn/activations/ReLU.h"
#include "nn/Serialization.h"
#include "optimizers/Adam.h"

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

std::pair<std::vector<int>, std::vector<int>> split_train_test_indices(const Tensor& y, float split_ratio = 0.8f) {
    int rows = y->size(0);
    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < rows; ++i) {
        int label = static_cast<int>(y->at({i, 0}));
        class_indices[label].push_back(i);
    }

    std::vector<int> train_idx;
    std::vector<int> test_idx;
    for (const auto &[label, indices] : class_indices) {
        int train_size = static_cast<int>(indices.size() * split_ratio);
        for (int i = 0; i < indices.size(); ++i) {
            if (i < train_size) {
                train_idx.push_back(indices[i]);
            } else {
                test_idx.push_back(indices[i]);
            }
        }
    }
    return {train_idx, test_idx};
}

std::pair<Tensor, Tensor> create_subset(const std::vector<int>& indices, const Tensor& features, const Tensor& labels) {
    int n = indices.size();
    int cols = features->size(1);
    std::vector<float> f_data(n * cols);
    std::vector<float> l_data(n);
    for (int i = 0; i < n; ++i) {
        int row = indices[i];
        for (int c = 0; c < cols; ++c) {
            f_data[i * cols + c] = features->at({row, c});
        }
        l_data[i] = labels->at({row, 0});
    }
    return std::make_pair(TensorImpl::from_data(f_data, {n, cols}, true), TensorImpl::from_data(l_data, {n}));
}

float calculate_accuracy(Linear& layer1, ReLU& relu, Linear& layer2, const Tensor& X, const Tensor& y) {
    Tensor h1 = layer1.forward(X);
    Tensor a1 = relu.forward(h1);
    Tensor logits = layer2.forward(a1);

    int correct = 0;
    int n = X->size(0);
    for (int i = 0; i < n; ++i) {
        float max_logit = -1e9f;
        int best_class = -1;
        for (int c = 0; c < 3; ++c) {
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
    std::string path = "Iris.csv";
    if (argc >= 2) {
        path = argv[1];
    }

    auto [X, y] = axon::data::load_csv(path, 5);

    auto [train_idx, test_idx] = split_train_test_indices(y, 0.8f);
    auto [X_train, y_train] = create_subset(train_idx, X, y);
    auto [X_test, y_test] = create_subset(test_idx, X, y);

    int cols = X->size(1);
    int hidden_layer_size = 32;

    Linear layer1(cols, hidden_layer_size);
    ReLU relu;
    Linear layer2(hidden_layer_size, 3);

    std::vector<Tensor> params;
    auto l1_p = layer1.parameters();
    auto l2_p = layer2.parameters();
    params.insert(params.end(), l1_p.begin(), l1_p.end());
    params.insert(params.end(), l2_p.begin(), l2_p.end());

    Adam optimizer(params, 0.01f);
    CrossEntropyLoss criterion;

    for (int epoch = 0; epoch < 400; ++epoch) {
        Tensor h1 = layer1.forward(X_train);
        Tensor a1 = relu.forward(h1);
        Tensor logits = layer2.forward(a1);

        Tensor loss = criterion.forward(logits, y_train);

        layer1.zero_grad();
        layer2.zero_grad();

        loss->backward();
        optimizer.step();

        if ((epoch + 1) % 50 == 0) {
            std::cout << "Epoch " << (epoch + 1) << " Loss: " << loss->at(0) << std::endl;
        }
    }

    float train_acc = calculate_accuracy(layer1, relu, layer2, X_train, y_train);
    std::cout << "Train accuracy: " << train_acc << "%" << std::endl;

    float test_acc = calculate_accuracy(layer1, relu, layer2, X_test, y_test);
    std::cout << "Test accuracy: " << test_acc << "%" << std::endl;

    std::string weights_path = "iris_model.bin";
    axon::save(params, weights_path);
    std::cout << "Saved model to " << weights_path << std::endl;

    Linear layer1_loaded(cols, hidden_layer_size);
    Linear layer2_loaded(hidden_layer_size, 3);

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
