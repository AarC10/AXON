#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H
#include <vector>

class Tensor {
public:
    Tensor();

    // Zero initialized ctor
    explicit Tensor(std::vector<int> shape, bool require_grad = false);

    // Data initialized ctor
    Tensor(std::vector<float> data, std::vector<int> shape, bool require_grad = false);

    // Copy / Move ctors
    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    Tensor& operator=(const Tensor& other);

    Tensor& operator=(Tensor&& other);

    ~Tensor() = default;

private:


};


#endif //AXON_TENSOR_H