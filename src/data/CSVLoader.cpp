#include "data/CSVLoader.h"

#include "core/TensorImpl.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace axon::data {

namespace {

std::string trim(const std::string &value) {
    std::size_t start = 0;
    std::size_t end = value.size();

    while (start < end && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }

    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(start, end - start);
}

std::vector<std::string> split_row(const std::string &line) {
    std::stringstream line_stream(line);
    std::string cell;
    std::vector<std::string> values;

    while (std::getline(line_stream, cell, ',')) {
        values.push_back(trim(cell));
    }

    return values;
}

float parse_float(const std::string &value, const std::string &error_context) {
    std::size_t parsed_chars = 0;
    float parsed_value = 0.0f;

    try {
        parsed_value = std::stof(value, &parsed_chars);
    } catch (const std::exception &) {
        throw std::invalid_argument(error_context + ": " + value);
    }

    if (parsed_chars != value.size()) {
        throw std::invalid_argument(error_context + ": " + value);
    }

    return parsed_value;
}

} // namespace

std::pair<Tensor, Tensor> load_csv(const std::string &path, int label_col, bool header) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + path);
    }

    std::string line;

    if (header) {
        std::getline(file, line);
    }

    std::vector<float> features;
    std::vector<float> labels;
    int expected_columns = -1;
    int feature_count = 0;
    int row_count = 0;
    std::unordered_map<std::string, float> label_encoding;
    float next_label_value = 0.0f;

    while (std::getline(file, line)) {
        std::vector<std::string> row = split_row(line);

        if (expected_columns == -1) {
            expected_columns = static_cast<int>(row.size());

            if (label_col < 0 || label_col >= expected_columns) {
                throw std::invalid_argument("label_col is out of range for CSV columns");
            }

            feature_count = expected_columns - 1;
        }

        if (static_cast<int>(row.size()) != expected_columns) {
            throw std::invalid_argument("CSV rows have inconsistent column counts");
        }

        for (int col = 0; col < expected_columns; ++col) {
            if (col == label_col) {
                try {
                    labels.push_back(parse_float(row[col], "Invalid numeric label"));
                } catch (const std::invalid_argument &) {
                    auto label_it = label_encoding.find(row[col]);

                    if (label_it == label_encoding.end()) {
                        label_it = label_encoding.emplace(row[col], next_label_value).first;
                        next_label_value += 1.0f;
                    }

                    labels.push_back(label_it->second);
                }
            } else {
                features.push_back(parse_float(row[col], "Invalid numeric feature"));
            }
        }

        ++row_count;
    }

    Tensor x = TensorImpl::from_data(features, {row_count, feature_count});
    Tensor y = TensorImpl::from_data(labels, {row_count, 1});

    return {x, y};
}

} // namespace axon::data
