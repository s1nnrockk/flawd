#include "flawd/nn.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace flawd {

Linear::Linear(std::size_t input_size, std::size_t output_size, std::uint32_t seed)
    : input_size_(input_size), output_size_(output_size) {
    if (input_size == 0 || output_size == 0)
        throw std::invalid_argument("Linear layer dimensions must be non-zero");
    std::mt19937 generator(seed);
    const float limit = std::sqrt(6.0F / static_cast<float>(input_size + output_size));
    std::uniform_real_distribution<float> distribution(-limit, limit);
    weights_.reserve(input_size * output_size);
    biases_.reserve(output_size);
    for (std::size_t i = 0; i < input_size * output_size; ++i)
        weights_.emplace_back(distribution(generator));
    for (std::size_t i = 0; i < output_size; ++i) biases_.emplace_back(0.0F);
}

ScalarVector Linear::forward(const ScalarVector& input) {
    if (input.size() != input_size_)
        throw std::invalid_argument("Linear layer input size mismatch");
    ScalarVector output;
    output.reserve(output_size_);
    for (std::size_t out = 0; out < output_size_; ++out) {
        Scalar sum = biases_[out];
        for (std::size_t in = 0; in < input_size_; ++in)
            sum = sum + input[in] * weights_[out * input_size_ + in];
        output.push_back(sum);
    }
    return output;
}
std::vector<Scalar> Linear::parameters() {
    std::vector<Scalar> result = weights_;
    result.insert(result.end(), biases_.begin(), biases_.end());
    return result;
}
std::size_t Linear::input_size() const noexcept { return input_size_; }
std::size_t Linear::output_size() const noexcept { return output_size_; }

ScalarVector ReLU::forward(const ScalarVector& input) {
    ScalarVector output;
    output.reserve(input.size());
    for (const auto& value : input) output.push_back(flawd::relu(value));
    return output;
}
std::vector<Scalar> ReLU::parameters() { return {}; }

ScalarVector Tanh::forward(const ScalarVector& input) {
    ScalarVector output;
    output.reserve(input.size());
    for (const auto& value : input) output.push_back(flawd::tanh(value));
    return output;
}
std::vector<Scalar> Tanh::parameters() { return {}; }

void Sequential::add(Layer& layer) { layers_.push_back(&layer); }
ScalarVector Sequential::forward(const ScalarVector& input) {
    ScalarVector result = input;
    for (auto* layer : layers_) result = layer->forward(result);
    return result;
}
std::vector<Scalar> Sequential::parameters() {
    std::vector<Scalar> result;
    for (auto* layer : layers_) {
        auto current = layer->parameters();
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

Scalar mean_squared_error(const ScalarVector& prediction, const ScalarVector& target) {
    if (prediction.empty() || prediction.size() != target.size())
        throw std::invalid_argument("MSE expects equally sized, non-empty vectors");
    Scalar total(0.0F);
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        const auto difference = prediction[i] - target[i];
        total = total + difference * difference;
    }
    return total / Scalar(static_cast<float>(prediction.size()));
}

SGD::SGD(std::vector<Scalar> parameters, float learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    set_learning_rate(learning_rate);
}
void SGD::zero_grad() noexcept { for (auto& parameter : parameters_) parameter.zero_grad(); }
void SGD::step() {
    for (auto& parameter : parameters_)
        parameter.set_value(parameter.value() - learning_rate_ * parameter.grad());
}
void SGD::set_learning_rate(float learning_rate) {
    if (!(learning_rate > 0.0F) || !std::isfinite(learning_rate))
        throw std::invalid_argument("Learning rate must be finite and positive");
    learning_rate_ = learning_rate;
}
float SGD::learning_rate() const noexcept { return learning_rate_; }

} // namespace flawd
