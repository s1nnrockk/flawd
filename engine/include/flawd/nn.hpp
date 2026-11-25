#pragma once

#include "flawd/autograd.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flawd {

using ScalarVector = std::vector<Scalar>;

class Layer {
public:
    virtual ~Layer() = default;
    virtual ScalarVector forward(const ScalarVector& input) = 0;
    virtual std::vector<Scalar> parameters() = 0;
};

class Linear final : public Layer {
public:
    Linear(std::size_t input_size, std::size_t output_size, std::uint32_t seed = 42);
    ScalarVector forward(const ScalarVector& input) override;
    std::vector<Scalar> parameters() override;
    std::size_t input_size() const noexcept;
    std::size_t output_size() const noexcept;

private:
    std::size_t input_size_;
    std::size_t output_size_;
    std::vector<Scalar> weights_;
    std::vector<Scalar> biases_;
};

class Embedding {
public:
    Embedding(std::size_t vocabulary_size, std::size_t embedding_size, std::uint32_t seed = 123);
    ScalarVector forward(std::size_t token_id) const;
    std::vector<Scalar> parameters();
    std::size_t vocabulary_size() const noexcept;
    std::size_t embedding_size() const noexcept;

private:
    std::size_t vocabulary_size_;
    std::size_t embedding_size_;
    std::vector<Scalar> weights_;
};

class ReLU final : public Layer {
public:
    ScalarVector forward(const ScalarVector& input) override;
    std::vector<Scalar> parameters() override;
};

class LayerNorm {
public:
    explicit LayerNorm(std::size_t feature_size, float epsilon = 1e-5F);
    ScalarVector forward(const ScalarVector& input);
    std::vector<Scalar> parameters();

private:
    std::size_t feature_size_;
    float epsilon_;
    std::vector<Scalar> gamma_;
    std::vector<Scalar> beta_;
};

ScalarVector softmax(const ScalarVector& logits);

class Tanh final : public Layer {
public:
    ScalarVector forward(const ScalarVector& input) override;
    std::vector<Scalar> parameters() override;
};

class Sequential {
public:
    void add(Layer& layer);
    ScalarVector forward(const ScalarVector& input);
    std::vector<Scalar> parameters();

private:
    std::vector<Layer*> layers_;
};

Scalar mean_squared_error(const ScalarVector& prediction, const ScalarVector& target);
Scalar cross_entropy_loss(const ScalarVector& logits, std::size_t target_index);

class SGD {
public:
    explicit SGD(std::vector<Scalar> parameters, float learning_rate = 0.01F);
    void zero_grad() noexcept;
    void step();
    void set_learning_rate(float learning_rate);
    float learning_rate() const noexcept;

private:
    std::vector<Scalar> parameters_;
    float learning_rate_;
};

class Adam {
public:
    explicit Adam(std::vector<Scalar> parameters, float learning_rate = 0.001F,
                  float beta1 = 0.9F, float beta2 = 0.999F, float epsilon = 1e-8F);
    void zero_grad() noexcept;
    void step();
    void set_learning_rate(float learning_rate);
    float learning_rate() const noexcept;

private:
    std::vector<Scalar> parameters_;
    std::vector<float> first_moment_;
    std::vector<float> second_moment_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    std::size_t step_count_ = 0;
};

} // namespace flawd
