BUILD_DIR  := build
CMAKE_OPTS := -DCMAKE_BUILD_TYPE=Release

.PHONY: all configure build tests examples iris_training mnist_training clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_OPTS)

build: configure
	cmake --build $(BUILD_DIR)

tests: build
	./$(BUILD_DIR)/axon_test

examples: build

iris_training: build
	./$(BUILD_DIR)/examples/iris_training/iris_training examples/data/Iris.csv

mnist_training: build
	./$(BUILD_DIR)/examples/mnist_training/mnist_training examples/data/mnist_train.csv examples/data/mnist_test.csv

clean:
	rm -rf $(BUILD_DIR)
