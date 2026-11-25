#include "flawd/trainer.hpp"

#include <stdexcept>

namespace flawd {

LanguageModelTrainer::LanguageModelTrainer(TransformerModel& model, float learning_rate)
    : model_(model), optimizer_(model.parameters(), learning_rate) {}

float LanguageModelTrainer::train_step(const std::vector<std::size_t>& input_tokens,
                                       std::size_t target_token) {
    if (input_tokens.empty()) throw std::invalid_argument("Training input cannot be empty");
    if (input_tokens.size() > model_.config().context_length)
        throw std::length_error("Training input exceeds model context length");
    if (target_token >= model_.config().vocabulary_size)
        throw std::out_of_range("Training target token is outside the vocabulary");

    optimizer_.zero_grad();
    const auto logits = model_.forward_last(input_tokens);
    const auto loss = cross_entropy_loss(logits, target_token);
    const float loss_value = loss.value();
    loss.backward();
    optimizer_.step();
    ++steps_;
    return loss_value;
}

TrainingStats LanguageModelTrainer::train(const std::vector<std::size_t>& token_stream,
                                          std::size_t epochs, std::size_t stride) {
    if (token_stream.size() < 2)
        throw std::invalid_argument("Training token stream must contain at least two tokens");
    if (epochs == 0 || stride == 0)
        throw std::invalid_argument("Epoch count and stride must be non-zero");
    for (const auto token : token_stream)
        if (token >= model_.config().vocabulary_size)
            throw std::out_of_range("Training stream contains a token outside the vocabulary");

    TrainingStats stats;
    double total_loss = 0.0;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t target_index = 1; target_index < token_stream.size(); target_index += stride) {
            const std::size_t start = target_index > model_.config().context_length
                ? target_index - model_.config().context_length : 0;
            std::vector<std::size_t> context(
                token_stream.begin() + static_cast<std::ptrdiff_t>(start),
                token_stream.begin() + static_cast<std::ptrdiff_t>(target_index));
            total_loss += train_step(context, token_stream[target_index]);
            ++stats.steps;
        }
    }
    stats.average_loss = total_loss / static_cast<double>(stats.steps);
    return stats;
}

TrainingStats LanguageModelTrainer::evaluate(const std::vector<std::size_t>& token_stream,
                                               std::size_t stride) {
    if (token_stream.size() < 2)
        throw std::invalid_argument("Evaluation token stream must contain at least two tokens");
    if (stride == 0) throw std::invalid_argument("Evaluation stride must be non-zero");
    for (const auto token : token_stream)
        if (token >= model_.config().vocabulary_size)
            throw std::out_of_range("Evaluation stream contains a token outside the vocabulary");

    TrainingStats stats;
    double total_loss = 0.0;
    for (std::size_t target_index = 1; target_index < token_stream.size(); target_index += stride) {
        const std::size_t start = target_index > model_.config().context_length
            ? target_index - model_.config().context_length : 0;
        const std::vector<std::size_t> context(
            token_stream.begin() + static_cast<std::ptrdiff_t>(start),
            token_stream.begin() + static_cast<std::ptrdiff_t>(target_index));
        const auto logits = model_.forward_last(context);
        total_loss += cross_entropy_loss(logits, token_stream[target_index]).value();
        ++stats.steps;
    }
    stats.average_loss = total_loss / static_cast<double>(stats.steps);
    return stats;
}

void LanguageModelTrainer::set_learning_rate(float learning_rate) {
    optimizer_.set_learning_rate(learning_rate);
}
std::size_t LanguageModelTrainer::steps() const noexcept { return steps_; }

} // namespace flawd
