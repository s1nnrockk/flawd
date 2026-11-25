#include "flawd/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace flawd {

std::size_t Tensor::element_count(const std::vector<std::size_t>& shape) {
    if (shape.empty()) throw std::invalid_argument("Tensor shape cannot be empty");
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension == 0) throw std::invalid_argument("Tensor dimensions must be non-zero");
        if (count > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::overflow_error("Tensor element count overflow");
        count *= dimension;
    }
    return count;
}

Tensor::Tensor(std::vector<std::size_t> shape, float fill_value)
    : shape_(std::move(shape)), values_(element_count(shape_), fill_value) {}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape_(std::move(shape)), values_(std::move(values)) {
    if (values_.size() != element_count(shape_))
        throw std::invalid_argument("Value count does not match tensor shape");
}

const std::vector<std::size_t>& Tensor::shape() const noexcept { return shape_; }
const std::vector<float>& Tensor::values() const noexcept { return values_; }
std::size_t Tensor::size() const noexcept { return values_.size(); }
std::size_t Tensor::rank() const noexcept { return shape_.size(); }

float& Tensor::operator[](std::size_t index) { return values_.at(index); }
const float& Tensor::operator[](std::size_t index) const { return values_.at(index); }

Tensor Tensor::from_values(std::initializer_list<float> values) {
    return Tensor({values.size()}, std::vector<float>(values));
}

namespace {
void require_same_shape(const Tensor& left, const Tensor& right, const char* operation) {
    if (left.shape() != right.shape())
        throw std::invalid_argument(std::string("Tensor shapes must match for ") + operation);
}
}

Tensor add(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right, "addition");
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] + right[i];
    return result;
}

Tensor subtract(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right, "subtraction");
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] - right[i];
    return result;
}

Tensor multiply(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right, "multiplication");
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] * right[i];
    return result;
}

Tensor scale(const Tensor& tensor, float factor) {
    Tensor result(tensor.shape());
    for (std::size_t i = 0; i < tensor.size(); ++i) result[i] = tensor[i] * factor;
    return result;
}

Tensor relu(const Tensor& tensor) {
    Tensor result(tensor.shape());
    for (std::size_t i = 0; i < tensor.size(); ++i) result[i] = std::max(0.0F, tensor[i]);
    return result;
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    if (left.rank() != 2 || right.rank() != 2)
        throw std::invalid_argument("matmul expects two rank-2 tensors");
    const auto rows = left.shape()[0];
    const auto inner = left.shape()[1];
    const auto right_inner = right.shape()[0];
    const auto columns = right.shape()[1];
    if (inner != right_inner)
        throw std::invalid_argument("matmul inner dimensions must match");

    Tensor result({rows, columns});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < columns; ++col) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < inner; ++k)
                sum += left[row * inner + k] * right[k * columns + col];
            result[row * columns + col] = sum;
        }
    }
    return result;
}

Tensor transpose_2d(const Tensor& tensor) {
    if (tensor.rank() != 2) throw std::invalid_argument("transpose_2d expects a rank-2 tensor");
    const auto rows = tensor.shape()[0];
    const auto columns = tensor.shape()[1];
    Tensor result({columns, rows});
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t col = 0; col < columns; ++col)
            result[col * rows + row] = tensor[row * columns + col];
    return result;
}

Tensor softmax(const Tensor& tensor) {
    if (tensor.rank() != 1) throw std::invalid_argument("softmax currently expects a rank-1 tensor");
    const auto max_value = *std::max_element(tensor.values().begin(), tensor.values().end());
    Tensor result(tensor.shape());
    double sum = 0.0;
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        result[i] = static_cast<float>(std::exp(static_cast<double>(tensor[i] - max_value)));
        sum += result[i];
    }
    if (sum == 0.0) throw std::runtime_error("softmax normalization failed");
    for (std::size_t i = 0; i < tensor.size(); ++i)
        result[i] = static_cast<float>(result[i] / sum);
    return result;
}

} // namespace flawd
