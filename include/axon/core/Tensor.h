#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H
#include <functional>
#include <memory>
#include <optional>
#include <vector>

class Tensor {
  public:
    /** @brief Constructs an empty tensor */
    Tensor();

    /**
     * @brief Constructs a zero-initialized tensor with the provided shape
     * @param shape Dimensions of the tensor
     * @param require_grad Whether gradients should be tracked
     */
    explicit Tensor(const std::vector<int>& shape, bool require_grad = false);

    /**
     * @brief Constructs a tensor from existing data and shape
     * @param data Flat data buffer
     * @param shape Dimensions of the tensor
     * @param require_grad Whether gradients should be tracked
     */
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool require_grad = false);

    /**
     * @brief Copy-constructs a tensor
     * @param other Tensor to copy from
     */
    Tensor(const Tensor &other);

    /**
     * @brief Move-constructs a tensor
     * @param other Tensor to move from
     */
    Tensor(Tensor &&other);

    /**
     * @brief Copy-assigns from another tensor
     * @param other Tensor to copy from
     * @return Reference to this tensor
     */
    Tensor &operator=(const Tensor &other);

    /**
     * @brief Move-assigns from another tensor
     * @param other Tensor to move from
     * @return Reference to this tensor
     */
    Tensor &operator=(Tensor &&other);

    /** @brief Destroys the tensor */
    ~Tensor() = default;

    // Static Factories

    /**
     * @brief Creates a tensor filled with zeros
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A zero-initialized tensor with the requested shape
     */
    static Tensor zeros(const std::vector<int>& shape, bool require_grad = false);

    /**
     * @brief Creates a tensor filled with ones
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A one-initialized tensor with the requested shape
     */
    static Tensor ones(const std::vector<int>& shape, bool require_grad = false);

    /**
     * @brief Creates a tensor filled with a value
     * @param shape Dimensions of the tensor
     * @param value Value to fill the tensor with
     * @param require_grad Whether to track gradients for this tensor
     * @return A uniformly initialized tensor with the requested shape
     */
    static Tensor full(const std::vector<int>& shape, float value, bool require_grad = false);

    /**
     * @brief Creates a tensor with values sampled from a normal distribution
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A randomly initialized tensor sampled from a normal distribution
     */
    static Tensor randn(const std::vector<int>& shape, bool require_grad = false);

    /**
     * @brief Creates a tensor with values sampled from a uniform distribution
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A randomly initialized tensor sampled from a uniform distribution
     */
    static Tensor rand(const std::vector<int>& shape, bool require_grad = false);

    /**
     * @brief Creates an identity matrix tensor of size n x n
     * @param n Number of rows and columns
     * @param require_grad Whether to track gradients for this tensor
     * @return An identity matrix tensor
     */
    static Tensor eye(int n, bool require_grad = false);

    /**
     * @brief Creates a 1D tensor with evenly spaced values in [start, stop)
     * @param start Start value of the sequence (inclusive)
     * @param stop End value of the sequence (exclusive)
     * @param step Increment between consecutive values
     * @param require_grad Whether to track gradients for this tensor
     * @return A 1D tensor containing the generated range
     */
    static Tensor arange(float start, float stop, float step = 1.0f, bool require_grad = false);

    // Shape
    /** @brief Returns the tensor shape */
    const std::vector<int> &get_shape() const;

    /** @brief Returns per-dimension strides in elements */
    const std::vector<int> &get_strides() const;

    /** @brief Returns the number of dimensions */
    int ndim() const;

    /** @brief Returns the number of elements in the tensor */
    int nelem() const;

    /**
     * @brief Get thje size of the tensor for a specified dimension
     * @param dim Dimension to query
     * @return Number of elements along dim
     */
    int size(int dim) const;

    /** @brief Returns whether this tensor tracks gradients */
    bool requires_grad() const;

    /**
     * @brief Enables or disables gradient tracking
     * @param require_grad New gradient tracking flag
     * @return True if the flag was applied
     */
    bool set_requires_grad(bool require_grad);

    /** @brief Returns true when storage layout is contiguous */
    bool is_contiguous();

    // Data Access
    /** @brief Returns a mutable pointer to underlying data */
    float *data();

    /** @brief Returns a const pointer to underlying data */
    const float *data() const;

    /**
     * @brief Returns the element value at a multidimensional index
     * @param idx Per-dimension index
     * @return Element value
     */
    float at(const std::vector<int> &idx) const;

    /**
     * @brief Returns a mutable reference at a multidimensional index
     * @param idx Per-dimension index
     * @return Mutable reference to the addressed element
     */
    float &at(const std::vector<int> &idx);

    /**
     * @brief Returns the element value at a flat index
     * @param idx Flat element index
     * @return Element value
     */
    float operator[](int idx) const;

    /**
     * @brief Returns a mutable reference at a flat index
     * @param idx Flat element index
     * @return Mutable reference to the addressed element
     */
    float &operator[](int idx);

    // Shape manip

    // Arithmetic Ops
    template<typename Operation>
    Tensor binary_op(const Tensor& rhs, Operation op) const;

    /** @brief Elementwise tensor addition */
    Tensor operator+(const Tensor &rhs) const;

    /** @brief Elementwise tensor subtraction */
    Tensor operator-(const Tensor &rhs) const;

    /** @brief Elementwise tensor multiplication */
    Tensor operator*(const Tensor &rhs) const;

    /** @brief Elementwise tensor division */
    Tensor operator/(const Tensor &rhs) const;

    /** @brief Elementwise unary negation */
    Tensor operator-() const;

    /** @brief Adds a scalar to every element */
    Tensor operator+(float scalar) const;

    /** @brief Subtracts a scalar from every element */
    Tensor operator-(float scalar) const;

    /** @brief Multiplies every element by a scalar */
    Tensor operator*(float scalar) const;

    /** @brief Divides every element by a scalar */
    Tensor operator/(float scalar) const;

    /** @brief In-place elementwise tensor addition */
    Tensor &operator+=(const Tensor &rhs);

    /** @brief In-place elementwise tensor subtraction */
    Tensor &operator-=(const Tensor &rhs);

    /** @brief In-place elementwise tensor multiplication */
    Tensor &operator*=(const Tensor &rhs);

    /** @brief In-place elementwise tensor division */
    Tensor &operator/=(const Tensor &rhs);

    /** @brief Adds a tensor to a lhs scalar */
    friend Tensor operator+(float scalar, const Tensor &tensor);

    /** @brief Subtracts a tensor from a lhs scalar   */
    friend Tensor operator-(float scalar, const Tensor &tensor);

    /** @brief Multiplies a lhs scalar by a tensor */
    friend Tensor operator*(float scalar, const Tensor &tensor);

    /** @brief Divides a lhs scalar by a tensor */
    friend Tensor operator/(float scalar, const Tensor &tensor);

    // Elementwise maffs
    /** @brief Elementwise exponential */
    Tensor exp() const;

    /** @brief Elementwise natural logarithm */
    Tensor log() const;

    /** @brief Elementwise square root */
    Tensor sqrt() const;

    /** @brief Elementwise absolute value */
    Tensor abs() const;
    /**
     * @brief Raises each element to a scalar exponent
     * @param exponent Scalar exponent
     * @return Result tensor
     */
    Tensor pow(float exponent) const;

    /**
     * @brief Raises each element to tensor-provided exponents
     * @param exp Elementwise exponents
     * @return Result tensor
     */
    Tensor pow(const Tensor &exp) const;

    /**
     * @brief Clamps values to a closed interval
     * @param min Lower bound
     * @param max Upper bound
     * @return Clipped tensor
     */
    Tensor clip(float min, float max) const;

    // COmparison ops
    /** @brief Elementwise equality comparison */
    Tensor operator==(const Tensor& rhs) const;

    /** @brief Elementwise inequality comparison */
    Tensor operator!=(const Tensor& rhs) const;

    /** @brief Elementwise LT comparison */
    Tensor operator< (const Tensor& rhs) const;

    /** @brief Elementwise LTE comparison */
    Tensor operator<=(const Tensor& rhs) const;

    /** @brief Elementwise GT comparison */
    Tensor operator> (const Tensor& rhs) const;

    /** @brief Elementwise GTE comparison */
    Tensor operator>=(const Tensor& rhs) const;

    // ACtivation functions

    // Reduction Ops

    // LinAlg

    // Autograd

    // Utils

    // Autograd handling
    using GradientFunc = std::function<void(const Tensor&)>;

    void set_gradient_func(GradientFunc func, const std::vector<std::shared_ptr<Tensor>>& inputs);

    bool get_is_leaf() const;

  private:
    std::shared_ptr<std::vector<float>> storage;
    int offset = 0;

    std::vector<int> shape;
    std::vector<int> strides;

    bool require_grad;
    bool is_leaf = true;

    std::shared_ptr<Tensor> grad;

    std::vector<std::shared_ptr<Tensor>> inputs;
    GradientFunc gradient_func;

    /**
     * @brief Converts a multidimensional index to a flat storage index
     * @param idx Per-dimension index
     * @return Flat index into storage
     */
    int flat_idnex(const std::vector<int> &idx) const;

    /** @brief Recomputes strides from the current shape */
    void compute_strides();

    /**
     * @brief Computes the flat storage index for a broadcasted access
     * @param flat Flat index into the broadcasted output
     * @param out_shape Shape of the broadcasted output
     * @param in_shape Shape of the input tensor being accessed
     * @return Flat index into the input tensor's storage corresponding to the broadcasted access
     */
    int broadcast_index(int flat, const std::vector<int>& out_shape, const std::vector<int>& in_shape) const;

    /**
     * @brief Computes broadcasted shape for two input shapes
     * @param shape_one First shape
     * @param shape_two Second shape
     * @return Broadcast-compatible shape
     */
    static std::vector<int> broadcast_shape(const std::vector<int> &shape_one, const std::vector<int> &shape_two);
};

#endif // AXON_TENSOR_H