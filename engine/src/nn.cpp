#include "flawd/nn.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
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


Embedding::Embedding(std::size_t vocabulary_size, std::size_t embedding_size, std::uint32_t seed)
    : vocabulary_size_(vocabulary_size), embedding_size_(embedding_size) {
    if (vocabulary_size == 0 || embedding_size == 0)
        throw std::invalid_argument("Embedding dimensions must be non-zero");
    std::mt19937 generator(seed);
    const float limit = 1.0F / std::sqrt(static_cast<float>(embedding_size));
    std::uniform_real_distribution<float> distribution(-limit, limit);
    weights_.reserve(vocabulary_size * embedding_size);
    for (std::size_t i = 0; i < vocabulary_size * embedding_size; ++i)
        weights_.emplace_back(distribution(generator));
}
ScalarVector Embedding::forward(std::size_t token_id) const {
    if (token_id >= vocabulary_size_) throw std::out_of_range("Embedding token id is out of range");
    ScalarVector result;
    result.reserve(embedding_size_);
    for (std::size_t i = 0; i < embedding_size_; ++i)
        result.push_back(weights_[token_id * embedding_size_ + i]);
    return result;
}
std::vector<Scalar> Embedding::parameters() { return weights_; }
std::size_t Embedding::vocabulary_size() const noexcept { return vocabulary_size_; }
std::size_t Embedding::embedding_size() const noexcept { return embedding_size_; }

ScalarVector ReLU::forward(const ScalarVector& input) {
    ScalarVector output;
    output.reserve(input.size());
    for (const auto& value : input) output.push_back(flawd::relu(value));
    return output;
}
std::vector<Scalar> ReLU::parameters() { return {}; }


LayerNorm::LayerNorm(std::size_t feature_size, float epsilon)
    : feature_size_(feature_size), epsilon_(epsilon) {
    if (feature_size == 0) throw std::invalid_argument("LayerNorm feature size must be non-zero");
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon))
        throw std::invalid_argument("LayerNorm epsilon must be finite and positive");
    gamma_.reserve(feature_size);
    beta_.reserve(feature_size);
    for (std::size_t i = 0; i < feature_size; ++i) {
        gamma_.emplace_back(1.0F);
        beta_.emplace_back(0.0F);
    }
}
ScalarVector LayerNorm::forward(const ScalarVector& input) {
    if (input.size() != feature_size_)
        throw std::invalid_argument("LayerNorm input size mismatch");
    Scalar mean(0.0F);
    for (const auto& value : input) mean = mean + value;
    mean = mean / Scalar(static_cast<float>(feature_size_));
    Scalar variance(0.0F);
    for (const auto& value : input) {
        const auto centered = value - mean;
        variance = variance + centered * centered;
    }
    variance = variance / Scalar(static_cast<float>(feature_size_));
    const auto inverse_std = Scalar(1.0F) / flawd::sqrt(variance + Scalar(epsilon_));
    ScalarVector output;
    output.reserve(feature_size_);
    for (std::size_t i = 0; i < feature_size_; ++i)
        output.push_back((input[i] - mean) * inverse_std * gamma_[i] + beta_[i]);
    return output;
}
std::vector<Scalar> LayerNorm::parameters() {
    std::vector<Scalar> result = gamma_;
    result.insert(result.end(), beta_.begin(), beta_.end());
    return result;
}

ScalarVector softmax(const ScalarVector& logits) {
    if (logits.empty()) throw std::invalid_argument("Softmax expects non-empty logits");
    float maximum = logits.front().value();
    for (const auto& value : logits) if (value.value() > maximum) maximum = value.value();
    Scalar denominator(0.0F);
    ScalarVector exponentials;
    exponentials.reserve(logits.size());
    for (const auto& value : logits) {
        exponentials.push_back(flawd::exp(value - Scalar(maximum)));
        denominator = denominator + exponentials.back();
    }
    ScalarVector probabilities;
    probabilities.reserve(logits.size());
    for (const auto& value : exponentials) probabilities.push_back(value / denominator);
    return probabilities;
}

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


Scalar cross_entropy_loss(const ScalarVector& logits, std::size_t target_index) {
    if (logits.empty()) throw std::invalid_argument("Cross entropy expects non-empty logits");
    if (target_index >= logits.size()) throw std::out_of_range("Cross entropy target index is out of range");
    float maximum = logits.front().value();
    for (const auto& value : logits)
        if (value.value() > maximum) maximum = value.value();
    Scalar exponential_sum(0.0F);
    for (const auto& value : logits)
        exponential_sum = exponential_sum + flawd::exp(value - Scalar(maximum));
    return Scalar(maximum) + flawd::log(exponential_sum) - logits[target_index];
}

Adam::Adam(std::vector<Scalar> parameters, float learning_rate, float beta1,
           float beta2, float epsilon)
    : parameters_(std::move(parameters)),
      first_moment_(parameters_.size(), 0.0F),
      second_moment_(parameters_.size(), 0.0F),
      learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    set_learning_rate(learning_rate);
    if (!(beta1 > 0.0F && beta1 < 1.0F) || !(beta2 > 0.0F && beta2 < 1.0F))
        throw std::invalid_argument("Adam beta values must be between zero and one");
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon))
        throw std::invalid_argument("Adam epsilon must be finite and positive");
}
void Adam::zero_grad() noexcept {
    for (auto& parameter : parameters_) parameter.zero_grad();
}
void Adam::step() {
    if (step_count_ == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Adam optimizer step counter overflow");
    for (const auto& parameter : parameters_) {
        if (!std::isfinite(parameter.value()) || !std::isfinite(parameter.grad()))
            throw std::runtime_error("Adam received a non-finite parameter or gradient");
    }
    ++step_count_;
    const float correction1 = 1.0F - std::pow(beta1_, static_cast<float>(step_count_));
    const float correction2 = 1.0F - std::pow(beta2_, static_cast<float>(step_count_));
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        const float gradient = parameters_[i].grad();
        first_moment_[i] = beta1_ * first_moment_[i] + (1.0F - beta1_) * gradient;
        second_moment_[i] = beta2_ * second_moment_[i] + (1.0F - beta2_) * gradient * gradient;
        const float corrected_first = first_moment_[i] / correction1;
        const float corrected_second = second_moment_[i] / correction2;
        const float update = learning_rate_ * corrected_first /
            (std::sqrt(corrected_second) + epsilon_);
        const float updated_value = parameters_[i].value() - update;
        if (!std::isfinite(update) || !std::isfinite(updated_value) ||
            !std::isfinite(first_moment_[i]) || !std::isfinite(second_moment_[i]))
            throw std::runtime_error("Adam update produced a non-finite value");
        parameters_[i].set_value(updated_value);
    }
}
void Adam::set_learning_rate(float learning_rate) {
    if (!(learning_rate > 0.0F) || !std::isfinite(learning_rate))
        throw std::invalid_argument("Learning rate must be finite and positive");
    learning_rate_ = learning_rate;
}
float Adam::learning_rate() const noexcept { return learning_rate_; }
std::uint64_t Adam::step_count() const noexcept {
    return static_cast<std::uint64_t>(step_count_);
}
AdamState Adam::state() const {
    AdamState result;
    result.learning_rate = learning_rate_;
    result.beta1 = beta1_;
    result.beta2 = beta2_;
    result.epsilon = epsilon_;
    result.step_count = static_cast<std::uint64_t>(step_count_);
    result.first_moment = first_moment_;
    result.second_moment = second_moment_;
    return result;
}
void Adam::load_state(const AdamState& state) {
    if (!(state.learning_rate > 0.0F) || !std::isfinite(state.learning_rate) ||
        !(state.beta1 > 0.0F && state.beta1 < 1.0F) ||
        !(state.beta2 > 0.0F && state.beta2 < 1.0F) ||
        !(state.epsilon > 0.0F) || !std::isfinite(state.epsilon))
        throw std::invalid_argument("Adam checkpoint contains invalid hyperparameters");
    if (state.step_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("Adam checkpoint step count exceeds this platform");
    if (state.first_moment.size() != parameters_.size() ||
        state.second_moment.size() != parameters_.size())
        throw std::invalid_argument("Adam checkpoint moment count does not match model parameters");
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        if (!std::isfinite(state.first_moment[index]) ||
            !std::isfinite(state.second_moment[index]) ||
            state.second_moment[index] < 0.0F)
            throw std::invalid_argument("Adam checkpoint contains invalid moment values");
    }
    learning_rate_ = state.learning_rate;
    beta1_ = state.beta1;
    beta2_ = state.beta2;
    epsilon_ = state.epsilon;
    step_count_ = static_cast<std::size_t>(state.step_count);
    first_moment_ = state.first_moment;
    second_moment_ = state.second_moment;
}
void Adam::reset_state() noexcept {
    std::fill(first_moment_.begin(), first_moment_.end(), 0.0F);
    std::fill(second_moment_.begin(), second_moment_.end(), 0.0F);
    step_count_ = 0;
}

} // namespace flawd
