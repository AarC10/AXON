#ifndef AXON_TENSOR_TPP
#define AXON_TENSOR_TPP

#include "core/Tensor.h"

template <typename Operation>
Tensor Tensor::binary_op(const Tensor& rhs, Operation op) const {
    auto out_shape = broadcast_shape(shape, rhs.shape);
    int n = 1;
    for (int d : out_shape) n *= d;

    Tensor out(out_shape, require_grad || rhs.require_grad);
    for (int i = 0; i < n; ++i) {
        int lhs_idx = offset + broadcast_index(i, out_shape, shape);
        int rhs_idx = rhs.offset + broadcast_index(i, out_shape, rhs.shape);
        (*out.storage)[i] = op((*storage)[lhs_idx], (*rhs.storage)[rhs_idx]);
    }
    return out;
}

#endif // AXON_TENSOR_TPP

