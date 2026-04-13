# AXON

Autograd eXecution and Optimization Network (AXON) is a machine learning library that implements:

- A tensor type with automatic differentiation
- Elementwise math and matrix multiplication
- Basic neural-network layers and activations
- Loss functions for regression and classification
- Common optimizers
- A Python binding built with `pybind11`

Components:

- `TensorImpl`: tensor storage, arithmetic, broadcasting, and autograd
- `Linear`: fully connected layer
- `ReLU`, `Sigmoid`: activation functions
- `MSELoss`, `CrossEntropyLoss`: loss functions
- `SGD`, `SGDWithMomentum`, `RMSProp`, `Adam`, `AdamW`: optimizers
- `CSVLoader`: CSV dataset loader
- `Serialization`: save/load parameter tensors

Build Targets:
- `axon_test`: C++ test binary
- `iris_training`: C++ Iris example
- `axon`: Python extension module when Python bindings are enabled

## Requirements

Toolchain:

- CMake 3.22+
- C++20 compiler

Optional dependencies:
- Python development environment for the Python module

Notes:
- Python bindings are enabled by default with `AXON_BUILD_PYTHON=ON`
- The build fetches `pybind11` during CMake configure requiring internet connection

## Build

## CMake
Configure:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build
```

This produces:

- `build/axon_test`
- `build/examples/iris_training/iris_training`
- `build/axon*.so` for the Python module when enabled

### Build without Python bindings

If you only want the C++ targets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAXON_BUILD_PYTHON=OFF
cmake --build build
```

### Build without OpenMP

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAXON_ENABLE_OPENMP=OFF
cmake --build build
```

## Make
A makefile supports wrapping around the cmake build

```bash
make build
```

Useful shortcuts:

- `make tests`
- `make iris_training`
- `make clean`

## Run

### Tests

with CMake output:

```bash
./build/axon_test
```

Make:

```bash
make tests
```

### C++ Iris example

```bash
./build/examples/iris_training/iris_training examples/Iris.csv
```

or

```bash
make iris_training
```

This trains a small classifier, prints training progress, reports train/test accuracy, and writes model weights to `iris_model.bin`.

### Python Iris example

Build first, then run:

```bash
python3 examples/iris_training_python/iris_training.py
```

The example script prepends `build/` to `sys.path`, so it expects the compiled module to exist there.
