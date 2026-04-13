#ifndef AXON_SERIALIZATION_H
#define AXON_SERIALIZATION_H

#include "core/TensorImpl.h"

#include <string>
#include <vector>

namespace axon {

/**
 * @brief Saves a list of parameter tensors to a binary file
 * @param params Vector of tensors to save
 * @param path File path to write to
 */
void save(const std::vector<Tensor> &params, const std::string &path);

/**
 * @brief Loads parameter data from a binary file into existing tensors
 * @param params Vector of tensors to load into (must match saved shapes)
 * @param path File path to read from
 */
void load(std::vector<Tensor> &params, const std::string &path);

} // namespace axon

#endif // AXON_SERIALIZATION_H
