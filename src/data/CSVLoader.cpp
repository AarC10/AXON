#include "data/CSVLoader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace axon::data {

namespace {

std::vector<float> parse_row(const std::string &line) {
    std::stringstream line_stream(line);
    std::string cell;
    std::vector<float> values;

    while (std::getline(line_stream, cell, ',')) {
        values.push_back(std::stof(cell));
    }

    return values;
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

    while (std::getline(file, line)) {
        std::vector<float> row = parse_row(line);

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
                labels.push_back(row[col]);
            } else {
                features.push_back(row[col]);
            }
        }

        ++row_count;
    }

    Tensor x(features, {row_count, feature_count});
    Tensor y(labels, {row_count, 1});

    return {x, y};
}

} // namespace axon::data
