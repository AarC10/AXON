#!/usr/bin/env python3

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.abspath(build_dir))

import axon


def split_train_test(X, y, split_ratio=0.8):
    """Stratified train/test split."""
    rows = y.size(0)

    class_indices = {}
    for i in range(rows):
        label = int(y[i])
        class_indices.setdefault(label, []).append(i)

    train_idx, test_idx = [], []
    for label, indices in class_indices.items():
        train_size = int(len(indices) * split_ratio)
        train_idx.extend(indices[:train_size])
        test_idx.extend(indices[train_size:])

    return train_idx, test_idx


def create_subset(indices, features, labels):
    n = len(indices)
    cols = features.size(1)

    f_data = []
    l_data = []
    for i in indices:
        for c in range(cols):
            f_data.append(features[i * cols + c])
        l_data.append(labels[i])

    X_sub = axon.Tensor.from_data(f_data, [n, cols], True)
    y_sub = axon.Tensor.from_data(l_data, [n])
    return X_sub, y_sub


def calculate_accuracy(layer1, relu, layer2, X, y):
    """Calculate classification accuracy."""
    h1 = layer1.forward(X)
    a1 = relu.forward(h1)
    logits = layer2.forward(a1)

    correct = 0
    n = X.size(0)
    num_classes = 3
    for i in range(n):
        best_class = 0
        best_logit = logits[i * num_classes]
        for c in range(1, num_classes):
            val = logits[i * num_classes + c]
            if val > best_logit:
                best_logit = val
                best_class = c
        if best_class == int(y[i]):
            correct += 1

    return correct * 100.0 / n


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Iris.csv")
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]

    X, y = axon.load_csv(csv_path, 5, True)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split
    train_idx, test_idx = split_train_test(X, y)
    X_train, y_train = create_subset(train_idx, X, y)
    X_test, y_test = create_subset(test_idx, X, y)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    cols = X.size(1)
    hidden_size = 32

    layer1 = axon.Linear(cols, hidden_size)
    relu = axon.ReLU()
    layer2 = axon.Linear(hidden_size, 3)

    params = layer1.parameters() + layer2.parameters()
    optimizer = axon.Adam(params, lr=0.01)
    criterion = axon.CrossEntropyLoss()

    for epoch in range(400):
        h1 = layer1.forward(X_train)
        a1 = relu.forward(h1)
        logits = layer2.forward(a1)

        loss = criterion.forward(logits, y_train)

        layer1.zero_grad()
        layer2.zero_grad()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d}  Loss: {loss.item():.6f}")

    train_acc = calculate_accuracy(layer1, relu, layer2, X_train, y_train)
    test_acc = calculate_accuracy(layer1, relu, layer2, X_test, y_test)
    print(f"Train accuracy: {train_acc:.1f}%")
    print(f"Test accuracy:  {test_acc:.1f}%")

    axon.save(params, "iris_model.bin")
    print("Saved model to iris_model.bin")

    layer1_loaded = axon.Linear(cols, hidden_size)
    layer2_loaded = axon.Linear(hidden_size, 3)
    loaded_params = layer1_loaded.parameters() + layer2_loaded.parameters()
    axon.load(loaded_params, "iris_model.bin")
    print("Loaded model from iris_model.bin")

    relu_loaded = axon.ReLU()
    loaded_acc = calculate_accuracy(layer1_loaded, relu_loaded, layer2_loaded, X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_acc:.1f}%")


if __name__ == "__main__":
    main()
