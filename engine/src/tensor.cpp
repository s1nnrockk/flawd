#include "flawd/tensor.hpp"

#include <stdexcept>
#include <utility>

namespace flawd {

std::size_t Tensor::element_count(const std::vector<std::size_t>& shape) {
    if (shape.empty()) throw std::invalid_argument("Tensor shape cannot be empty");
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension == 0) throw std::invalid_argument("Tensor dimensions must be non-zero");
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

float& Tensor::operator[](std::size_t index) { return values_.at(index); }
const float& Tensor::operator[](std::size_t index) const { return values_.at(index); }

Tensor Tensor::from_values(std::initializer_list<float> values) {
    return Tensor({values.size()}, std::vector<float>(values));
}

Tensor add(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape())
        throw std::invalid_argument("Tensor shapes must match for addition");
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] + right[i];
    return result;
}

Tensor multiply(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape())
        throw std::invalid_argument("Tensor shapes must match for multiplication");
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] * right[i];
    return result;
}

Tensor scale(const Tensor& tensor, float factor) {
    Tensor result(tensor.shape());
    for (std::size_t i = 0; i < tensor.size(); ++i) result[i] = tensor[i] * factor;
    return result;
}

} // namespace flawd
