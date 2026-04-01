#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H
#include <memory>
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

    // Static Factories

    /**
     * @brief Creates tensor filled with zeros
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A zero-initialized tensor with the requested shape
     */
    static Tensor zeros(std::vector<int> shape, bool require_grad = false);

    /**
     * @brief Creates tensor filled with ones
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A one-initialized tensor with the requested shape
     */
    static Tensor ones(std::vector<int> shape, bool require_grad = false);

    /**
     * @brief Creates tensor filled with a uniform default value
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A uniformly initialized tensor with the requested shape
     */
    static Tensor full(std::vector<int> shape, bool require_grad = false);

    /**
     * @brief Creates tensor with values sampled from a normal distribution
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A randomly initialized tensor sampled from a normal distribution
     */
    static Tensor randn(std::vector<int> shape, bool require_grad = false);

    /**
     * @brief Creates tensor with values sampled from a uniform distribution
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A randomly initialized tensor sampled from a uniform distribution
     */
    static Tensor rand(std::vector<int> shape, bool require_grad = false);

    /**
     * @brief Creates an identity matrix tensor of size n x n
     * @param n Number of rows and columns
     * @param require_grad Whether to track gradients for this tensor
     * @return An identity matrix tensor
     */
    static Tensor eye(int n, bool require_grad = false);

    /**
     * @brief Creates a 1D tensor with evenly spaced values in [start, stop)
     * @param start Start value of the sequence. Inclusive
     * @param stop End value of the sequence. Exclusive
     * @param step Increment between consecutive values
     * @param require_grad Whether to track gradients for this tensor
     * @return A 1D tensor containing the generated range
     */
    static Tensor arange(float start, float stop, float step = 1.0f, bool require_grad = false);

    // Data Access


    // Shape manip

    // Arithmetic Ops

    // COmparison ops

    // Elementwise maffs

    // ACtivation functions

    // Reduction Ops

    // LinAlg

    // Autograd

    // Utils





private:
    std::shared_ptr<std::vector<float>> data;
    int offset = 0;

    std::vector<int> shape;
    std::vector<int> stride;

    bool require_grad;
    bool is_leaf = true;


    std::shared_ptr<Tensor> grad;


    std::vector<std::shared_ptr<Tensor>> inputs;


    int flat_idnex(const std::vector<int>& idx) const;

    void compute_strides();

    static std::vector<int> broadcast_shape(const std::vector<int>& shape_one, const std::vector<int>& shape_two);
};


#endif //AXON_TENSOR_H