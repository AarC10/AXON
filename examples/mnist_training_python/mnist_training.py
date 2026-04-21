#!/usr/bin/env python3

import os
import random
import sys

build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.abspath(build_dir))

import axon

NUM_CLASSES = 10


def load_and_normalize(path):
    X_raw, y_raw = axon.load_csv(path, 0, True)
    rows = X_raw.size(0)
    cols = X_raw.size(1)

    x_data = [X_raw[i * cols + c] / 255.0 for i in range(rows) for c in range(cols)]
    y_data = [y_raw[i] for i in range(rows)]

    X = axon.Tensor.from_data(x_data, [rows, cols], True)
    y = axon.Tensor.from_data(y_data, [rows])
    return X, y


def gather_batch(indices, X, y):
    n = len(indices)
    cols = X.size(1)

    f_data = []
    l_data = []
    for i in indices:
        for c in range(cols):
            f_data.append(X[i * cols + c])
        l_data.append(y[i])

    Xb = axon.Tensor.from_data(f_data, [n, cols], True)
    yb = axon.Tensor.from_data(l_data, [n])
    return Xb, yb


def calculate_accuracy(layer1, relu, layer2, X, y):
    h1 = layer1.forward(X)
    a1 = relu.forward(h1)
    logits = layer2.forward(a1)

    correct = 0
    n = X.size(0)
    for i in range(n):
        best_class = 0
        best_logit = logits[i * NUM_CLASSES]
        for c in range(1, NUM_CLASSES):
            val = logits[i * NUM_CLASSES + c]
            if val > best_logit:
                best_logit = val
                best_class = c
        if best_class == int(y[i]):
            correct += 1

    return correct * 100.0 / n


def calculate_loss(layer1, relu, layer2, criterion, X, y):
    h1 = layer1.forward(X)
    a1 = relu.forward(h1)
    logits = layer2.forward(a1)
    return criterion.forward(logits, y).item()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(data_dir, "mnist_train.csv")
    test_path = os.path.join(data_dir, "mnist_test.csv")
    if len(sys.argv) >= 2:
        train_path = sys.argv[1]
    if len(sys.argv) >= 3:
        test_path = sys.argv[2]

    X_train, y_train = load_and_normalize(train_path)
    X_test, y_test = load_and_normalize(test_path)

    cols = X_train.size(1)
    train_n = X_train.size(0)
    print(f"Train: {train_n} samples, Test: {X_test.size(0)} samples, {cols} features")

    hidden_size = 128
    layer1 = axon.Linear(cols, hidden_size)
    relu = axon.ReLU()
    layer2 = axon.Linear(hidden_size, NUM_CLASSES)

    params = layer1.parameters() + layer2.parameters()
    optimizer = axon.Adam(params, lr=0.001)
    criterion = axon.CrossEntropyLoss()

    batch_size = 64
    num_epochs = 10
    rng = random.Random(0)

    losses = []
    train_accs = []
    test_accs = []

    initial_train_loss = calculate_loss(layer1, relu, layer2, criterion, X_train, y_train)
    initial_test_loss = calculate_loss(layer1, relu, layer2, criterion, X_test, y_test)
    initial_train_acc = calculate_accuracy(layer1, relu, layer2, X_train, y_train)
    initial_test_acc = calculate_accuracy(layer1, relu, layer2, X_test, y_test)
    print(f"Before training  Train loss: {initial_train_loss:.6f}  "
          f"Train: {initial_train_acc:.2f}%  "
          f"Test loss: {initial_test_loss:.6f}  Test: {initial_test_acc:.2f}%")

    for epoch in range(num_epochs):
        perm = list(range(train_n))
        rng.shuffle(perm)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, train_n - batch_size + 1, batch_size):
            batch_idx = perm[start:start + batch_size]
            Xb, yb = gather_batch(batch_idx, X_train, y_train)

            h1 = layer1.forward(Xb)
            a1 = relu.forward(h1)
            logits = layer2.forward(a1)

            loss = criterion.forward(logits, yb)

            layer1.zero_grad()
            layer2.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        train_acc = calculate_accuracy(layer1, relu, layer2, X_train, y_train)
        test_acc = calculate_accuracy(layer1, relu, layer2, X_test, y_test)

        losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1:2d}  Loss: {avg_loss:.6f}  "
              f"Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")

    print(f"Final train accuracy: {train_accs[-1]:.2f}%")
    print(f"Final test accuracy:  {test_accs[-1]:.2f}%")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
    else:
        epochs = range(1, num_epochs + 1)
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

        ax_loss.plot(epochs, losses, color="tab:red")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss")
        ax_loss.grid(True, alpha=0.3)

        ax_acc.plot(epochs, train_accs, label="Train", color="tab:blue")
        ax_acc.plot(epochs, test_accs, label="Test", color="tab:green")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy")
        ax_acc.set_ylim(0, 105)
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = "mnist_curves.png"
        fig.savefig(out_path, dpi=120)
        print(f"Saved curves to {out_path}")
        plt.show()

    axon.save(params, "mnist_model.bin")
    print("Saved model to mnist_model.bin")

    layer1_loaded = axon.Linear(cols, hidden_size)
    layer2_loaded = axon.Linear(hidden_size, NUM_CLASSES)
    loaded_params = layer1_loaded.parameters() + layer2_loaded.parameters()
    axon.load(loaded_params, "mnist_model.bin")
    print("Loaded model from mnist_model.bin")

    relu_loaded = axon.ReLU()
    loaded_acc = calculate_accuracy(layer1_loaded, relu_loaded, layer2_loaded, X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_acc:.2f}%")


if __name__ == "__main__":
    main()
