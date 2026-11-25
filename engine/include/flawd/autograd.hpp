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
    friend Scalar operator+(const Scalar&, const Scalar&);
    friend Scalar operator-(const Scalar&, const Scalar&);
    friend Scalar operator*(const Scalar&, const Scalar&);
    friend Scalar operator/(const Scalar&, const Scalar&);
    friend Scalar tanh(const Scalar&);
    friend Scalar relu(const Scalar&);
    friend Scalar exp(const Scalar&);
    friend Scalar sqrt(const Scalar&);
    friend Scalar log(const Scalar&);
private:
    struct Node;
    explicit Scalar(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_;
};
Scalar tanh(const Scalar&);
Scalar relu(const Scalar&);
Scalar exp(const Scalar&);
Scalar sqrt(const Scalar&);
Scalar log(const Scalar&);
} // namespace flawd
