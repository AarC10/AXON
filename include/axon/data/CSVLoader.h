#ifndef AXON_CSVLOADER_H
#define AXON_CSVLOADER_H

#include "core/Tensor.h"

#include <string>
#include <utility>

namespace axon::data {

std::pair<Tensor, Tensor> load_csv(const std::string &path, int label_col, bool header = true);

} // namespace axon::data

#endif // AXON_CSVLOADER_H
