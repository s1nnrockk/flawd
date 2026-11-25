#pragma once

#include <memory>
#include <vector>

namespace flawd {

class Scalar {
public:
    explicit Scalar(float value = 0.0F);
    float value() const noexcept;
    float grad() const noexcept;
    void set_value(float value);
    void zero_grad() noexcept;
    void backward() const;

    friend Scalar operator+(const Scalar& left, const Scalar& right);
    friend Scalar operator-(const Scalar& left, const Scalar& right);
    friend Scalar operator*(const Scalar& left, const Scalar& right);
    friend Scalar operator/(const Scalar& left, const Scalar& right);
    friend Scalar tanh(const Scalar& input);
    friend Scalar relu(const Scalar& input);
    friend Scalar exp(const Scalar& input);
    friend Scalar sqrt(const Scalar& input);
    friend Scalar log(const Scalar& input);

private:
    struct Node;
    explicit Scalar(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_;
    friend class SGD;
};

Scalar tanh(const Scalar& input);
Scalar relu(const Scalar& input);
Scalar exp(const Scalar& input);
Scalar sqrt(const Scalar& input);

} // namespace flawd
