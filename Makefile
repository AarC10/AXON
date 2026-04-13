BUILD_DIR  := build
CMAKE_OPTS := -DCMAKE_BUILD_TYPE=Release

.PHONY: all configure build tests examples iris_training clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_OPTS)

build: configure
	cmake --build $(BUILD_DIR)

tests: build
	./$(BUILD_DIR)/axon_test

examples: build

iris_training: build
	./$(BUILD_DIR)/examples/iris_training/iris_training examples/Iris.csv

clean:
	rm -rf $(BUILD_DIR)
