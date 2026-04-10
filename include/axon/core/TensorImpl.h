#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H
#include <functional>
#include <memory>
#include <vector>

class TensorImpl;
using Tensor = std::shared_ptr<TensorImpl>;

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
  public:

    /** @brief Constructs an empty tensor */
    TensorImpl() = default;

    /**
     * @brief Constructs a zero-initialized tensor with the provided shape
     * @param shape Dimensions of the tensor
     * @param require_grad Whether gradients should be tracked
     */
    explicit TensorImpl(const std::vector<int> &shape, bool require_grad = false);

    /**
     * @brief Constructs a tensor from existing data and shape
     * @param data Flat data buffer
     * @param shape Dimensions of the tensor
     * @param require_grad Whether gradients should be tracked
     */
    TensorImpl(const std::vector<float> &data, const std::vector<int> &shape, bool require_grad = false);

    /**
     * @brief Copy-constructs a tensor
     * @param other TensorImpl to copy from
     */
    TensorImpl(const TensorImpl &other);

    /**
     * @brief Move-constructs a tensor
     * @param other TensorImpl to move from
     */
    TensorImpl(TensorImpl &&other);

    /**
     * @brief Copy-assigns from another tensor
     * @param other TensorImpl to copy from
     * @return Reference to this tensor
     */
    TensorImpl &operator=(const TensorImpl &other);

    /**
     * @brief Move-assigns from another tensor
     * @param other TensorImpl to move from
     * @return Reference to this tensor
     */
    TensorImpl &operator=(TensorImpl &&other);

    /** @brief Destroys the tensor */
    ~TensorImpl() = default;

    // ==================================
    // ======== Static Factories ========
    // ==================================

    /**
     * @brief Creates a tensor filled with zeros
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A zero-initialized tensor with the requested shape
     */
    static TensorImpl zeros(const std::vector<int> &shape, bool require_grad = false);

    /**
     * @brief Creates a tensor filled with ones
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A one-initialized tensor with the requested shape
     */
    static TensorImpl ones(const std::vector<int> &shape, bool require_grad = false);

    /**
     * @brief Creates a tensor filled with a value
     * @param shape Dimensions of the tensor
     * @param value Value to fill the tensor with
     * @param require_grad Whether to track gradients for this tensor
     * @return A uniformly initialized tensor with the requested shape
     */
    static TensorImpl full(const std::vector<int> &shape, float value, bool require_grad = false);

    /**
     * @brief Creates a tensor with values sampled from a standard normal distribution
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A randomly initialized tensor sampled from a normal distribution
     */
    static TensorImpl randn(const std::vector<int> &shape, bool require_grad = false);

    /**
     * @brief Creates a tensor with values sampled from a uniform distribution over [0, 1)
     * @param shape Dimensions of the tensor
     * @param require_grad Whether to track gradients for this tensor
     * @return A randomly initialized tensor sampled from a uniform distribution
     */
    static TensorImpl rand(const std::vector<int> &shape, bool require_grad = false);

    /**
     * @brief Creates an identity matrix tensor of size n x n
     * @param n Number of rows and columns
     * @param require_grad Whether to track gradients for this tensor
     * @return An identity matrix tensor
     */
    static TensorImpl eye(int n, bool require_grad = false);

    /**
     * @brief Creates a 1D tensor with evenly spaced values in [start, stop)
     * @param start Start value of the sequence (inclusive)
     * @param stop End value of the sequence (exclusive)
     * @param step Increment between consecutive values
     * @param require_grad Whether to track gradients for this tensor
     * @return A 1D tensor containing the generated range
     */
    static TensorImpl arange(float start, float stop, float step = 1.0f, bool require_grad = false);
    // ==================================
    // ============== Shape =============
    // ==================================

    /** @brief Returns the tensor shape */
    const std::vector<int> &get_shape() const;

    /** @brief Returns per-dimension strides in elements */
    const std::vector<int> &get_strides() const;

    /** @brief Returns the number of dimensions */
    int ndim() const;

    /** @brief Returns the number of elements in the tensor */
    int nelem() const;

    /**
     * @brief Get the size of the tensor for a specified dimension
     * @param dim Dimension to query
     * @return Number of elements along dim
     */
    int size(int dim) const;

    /** @brief Returns whether this tensor tracks gradients */
    bool get_require_grad() const;

    /**
     * @brief Enables or disables gradient tracking
     * @param require_grad New gradient tracking flag
     */
    void set_require_grad(bool require_grad);

    /** @brief Returns true when storage layout is contiguous */
    bool is_contiguous() const;

    // ==================================
    // ========== Data Access ===========
    // ==================================

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
    // ==================================
    // ======= Shape Manipulation =======
    // ==================================

    // ==================================
    // ====== Arithmetic Operations =====
    // ==================================

    template <typename Operation>
    TensorImpl binary_op(const TensorImpl &rhs, Operation op) const;

    /** @brief Elementwise tensor addition */
    TensorImpl operator+(const TensorImpl& rhs) const;

/** @brief Elementwise tensor addition */
    friend Tensor operator+(Tensor lhs_data, Tensor rhs_data);

    /** @brief Elementwise tensor subtraction */
    TensorImpl operator-(const TensorImpl &rhs) const;

    /** @brief Elementwise tensor multiplication */
    TensorImpl operator*(const TensorImpl &rhs) const;

    /** @brief Elementwise tensor division */
    TensorImpl operator/(const TensorImpl &rhs) const;

    /** @brief Elementwise unary negation */
    TensorImpl operator-() const;

    /** @brief Adds a scalar to every element */
    TensorImpl operator+(float scalar) const;

    /** @brief Subtracts a scalar from every element */
    TensorImpl operator-(float scalar) const;

    /** @brief Multiplies every element by a scalar */
    TensorImpl operator*(float scalar) const;

    /** @brief Divides every element by a scalar */
    TensorImpl operator/(float scalar) const;

    /** @brief In-place elementwise tensor addition */
    TensorImpl &operator+=(const TensorImpl &rhs);

    /** @brief In-place elementwise tensor subtraction */
    TensorImpl &operator-=(const TensorImpl &rhs);

    /** @brief In-place elementwise tensor multiplication */
    TensorImpl &operator*=(const TensorImpl &rhs);

    /** @brief In-place elementwise tensor division */
    TensorImpl &operator/=(const TensorImpl &rhs);

    /** @brief Adds a tensor to a lhs scalar */
    friend TensorImpl operator+(float scalar, const TensorImpl &tensor);

    /** @brief Subtracts a tensor from a lhs scalar   */
    friend TensorImpl operator-(float scalar, const TensorImpl &tensor);

    /** @brief Multiplies a lhs scalar by a tensor */
    friend TensorImpl operator*(float scalar, const TensorImpl &tensor);

    /** @brief Divides a lhs scalar by a tensor */
    friend TensorImpl operator/(float scalar, const TensorImpl &tensor);

    // ==================================
    // ======== Elementwise Math ========
    // ==================================

    /** @brief Elementwise exponential */
    TensorImpl exp() const;

    /** @brief Elementwise natural logarithm */
    TensorImpl log() const;

    /** @brief Elementwise square root */
    TensorImpl sqrt() const;

    /** @brief Elementwise absolute value */
    TensorImpl abs() const;

    /**
     * @brief Raises each element to a scalar exponent
     * @param exponent Scalar exponent
     * @return Result tensor
     */
    TensorImpl pow(float exponent) const;

    /**
     * @brief Raises each element to tensor-provided exponents
     * @param exp Elementwise exponents
     * @return Result tensor
     */
    TensorImpl pow(const TensorImpl &exp) const;

    /**
     * @brief Clamps values to a closed interval
     * @param min Lower bound
     * @param max Upper bound
     * @return Clipped tensor
     */
    TensorImpl clip(float min, float max) const;

    // ======== Comparison Operations ========
    /** @brief Elementwise equality comparison */
    TensorImpl operator==(const TensorImpl &rhs) const;

    /** @brief Elementwise inequality comparison */
    TensorImpl operator!=(const TensorImpl &rhs) const;

    /** @brief Elementwise LT comparison */
    TensorImpl operator<(const TensorImpl &rhs) const;

    /** @brief Elementwise LTE comparison */
    TensorImpl operator<=(const TensorImpl &rhs) const;

    /** @brief Elementwise GT comparison */
    TensorImpl operator>(const TensorImpl &rhs) const;

    /** @brief Elementwise GTE comparison */
    TensorImpl operator>=(const TensorImpl &rhs) const;

    // ==================================
    // ====== Activation Functions ======
    // ==================================

    // ==================================
    // ====== Reduction Operations ======
    // ==================================

    // ==================================
    // ========= Linear Algebra =========
    // ==================================

    // ==================================
    // ============ Autograd ============
    // ==================================

    // TODO: Matt complete these functions
    /**
     * @brief Runs backpropagation from this tensor through the computation graph,
     *        accumulating gradients into all leaf tensors that require grad
     */
    void backward();

    /**
     * Get the gradient accumulated tensor
     * @return Reference to gradient tensor
     */
    TensorImpl& grad();

    /**
     * @brief Get a const ref to the gradient accumulated tensor
     * @return Const ref to gradient tensor
     */
    const TensorImpl& grad() const;

    /**
     * @brief Get whether the tensor has a gradient
     * @return True if the tensor has a gradient, false otherwise
     */
    bool has_grad() const;

    /** @brief Zeros out the gradient for this tensor */
    void zero_grad();

    // ==================================
    // =========== Utilities ===========
    // ==================================

    // ==================================
    // ======== Autograd Handling ========
    // ==================================

    using GradientFunc = std::function<void(const TensorImpl &)>;
    using GradientFuncPtr = std::function<void(std::shared_ptr<TensorImpl>)>;

    void set_gradient_func(GradientFunc func, const std::vector<std::shared_ptr<TensorImpl>> &inputs);

    bool get_is_leaf() const;

  private:
    std::shared_ptr<std::vector<float>> storage;
    int offset = 0;

    std::vector<int> shape;
    std::vector<int> strides;

    bool require_grad = false;
    bool is_leaf = true;

    Tensor gradient;
    uint backprop_dep_count = 0;

    std::vector<Tensor> inputs;
    GradientFunc gradient_func;
    GradientFuncPtr gradient_func_ptr;

    /**
     * @brief Converts a multidimensional index to a flat storage index
     * @param idx Per-dimension index
     * @return Flat index into storage
     */
    int flat_index(const std::vector<int> &idx) const;

    /** @brief Recomputes strides from the current shape */
    void compute_strides();

    /** @brief Strips all autograd state so the tensor behaves as a plain leaf */
    void detach_grad_state();

    /**
     * @brief Computes the flat storage index for a broadcasted access
     * @param flat Flat index into the broadcasted output
     * @param out_shape Shape of the broadcasted output
     * @param in_shape Shape of the input tensor being accessed
     * @return Flat index into the input tensor's storage corresponding to the broadcasted access
     */
    int broadcast_index(int flat, const std::vector<int> &out_shape, const std::vector<int> &in_shape) const;

    /**
     * @brief Computes broadcasted shape for two input shapes
     * @param shape_one First shape
     * @param shape_two Second shape
     * @return Broadcast-compatible shape
     */
    static std::vector<int> broadcast_shape(const std::vector<int> &shape_one, const std::vector<int> &shape_two);
};

#include "core/TensorImpl.tpp"


#endif // AXON_TENSOR_H
