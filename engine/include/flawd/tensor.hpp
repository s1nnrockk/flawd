#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace flawd {

class Tensor {
public:
    explicit Tensor(std::vector<std::size_t> shape, float fill_value = 0.0F);
    Tensor(std::vector<std::size_t> shape, std::vector<float> values);

    const std::vector<std::size_t>& shape() const noexcept;
    const std::vector<float>& values() const noexcept;
    std::size_t size() const noexcept;
    std::size_t rank() const noexcept;

    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    static Tensor from_values(std::initializer_list<float> values);

private:
    static std::size_t element_count(const std::vector<std::size_t>& shape);
    std::vector<std::size_t> shape_;
    std::vector<float> values_;
};

Tensor add(const Tensor& left, const Tensor& right);
Tensor subtract(const Tensor& left, const Tensor& right);
Tensor multiply(const Tensor& left, const Tensor& right);
Tensor scale(const Tensor& tensor, float factor);
Tensor relu(const Tensor& tensor);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor transpose_2d(const Tensor& tensor);
Tensor softmax(const Tensor& tensor);

} // namespace flawd
