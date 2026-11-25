#include "flawd/trainer.hpp"

#include <stdexcept>

namespace flawd {
namespace {
std::size_t context_start(const std::vector<std::size_t>& tokens,
                          std::size_t target_index,
                          std::size_t context_length,
                          std::size_t vocabulary_size) {
    std::size_t start = target_index > context_length
        ? target_index - context_length : 0;
    // FLAWD's byte tokenizer reserves token 257 for BOS. Never let a
    // training example attend to the previous record across this boundary.
    if (vocabulary_size > 257) {
        for (std::size_t index = target_index; index > start; --index) {
            if (tokens[index - 1] == 257) {
                start = index - 1;
                break;
            }
        }
    }
    return start;
}
} // namespace

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
                                          std::size_t epochs, std::size_t stride,
                                          std::size_t gradient_accumulation) {
    return train(token_stream, std::vector<std::uint8_t>(token_stream.size(), 1),
                 epochs, stride, gradient_accumulation);
}

TrainingStats LanguageModelTrainer::train(const std::vector<std::size_t>& token_stream,
                                          const std::vector<std::uint8_t>& loss_mask,
                                          std::size_t epochs, std::size_t stride,
                                          std::size_t gradient_accumulation) {
    if (token_stream.size() < 2)
        throw std::invalid_argument("Training token stream must contain at least two tokens");
    if (loss_mask.size() != token_stream.size())
        throw std::invalid_argument("Training loss mask size must match token stream");
    if (epochs == 0 || stride == 0 || gradient_accumulation == 0)
        throw std::invalid_argument("Epoch count, stride, and gradient accumulation must be non-zero");
    for (const auto token : token_stream)
        if (token >= model_.config().vocabulary_size)
            throw std::out_of_range("Training stream contains a token outside the vocabulary");
    for (const auto flag : loss_mask)
        if (flag > 1) throw std::invalid_argument("Training loss mask values must be 0 or 1");

    TrainingStats stats;
    double total_loss = 0.0;
    auto model_parameters = model_.parameters();
    std::size_t accumulated = 0;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t target_index = 1; target_index < token_stream.size(); target_index += stride) {
            if (loss_mask[target_index] == 0) continue;
            if (accumulated == 0) optimizer_.zero_grad();
            const std::size_t start = context_start(token_stream, target_index,
                                                    model_.config().context_length,
                                                    model_.config().vocabulary_size);
            std::vector<std::size_t> context(
                token_stream.begin() + static_cast<std::ptrdiff_t>(start),
                token_stream.begin() + static_cast<std::ptrdiff_t>(target_index));
            const auto logits = model_.forward_last(context);
            const auto loss = cross_entropy_loss(logits, token_stream[target_index]);
            total_loss += loss.value();
            loss.backward();
            ++accumulated;
            ++stats.steps;

            if (accumulated == gradient_accumulation) {
                const float scale = 1.0F / static_cast<float>(accumulated);
                for (auto& parameter : model_parameters) parameter.scale_grad(scale);
                optimizer_.step();
                ++steps_;
                accumulated = 0;
            }
        }
    }
    if (accumulated > 0) {
        const float scale = 1.0F / static_cast<float>(accumulated);
        for (auto& parameter : model_parameters) parameter.scale_grad(scale);
        optimizer_.step();
        ++steps_;
    }
    if (stats.steps == 0)
        throw std::invalid_argument("Training loss mask selects no next-token targets");
    stats.average_loss = total_loss / static_cast<double>(stats.steps);
    return stats;
}

TrainingStats LanguageModelTrainer::evaluate(const std::vector<std::size_t>& token_stream,
                                               std::size_t stride) {
    return evaluate(token_stream, std::vector<std::uint8_t>(token_stream.size(), 1), stride);
}

TrainingStats LanguageModelTrainer::evaluate(const std::vector<std::size_t>& token_stream,
                                               const std::vector<std::uint8_t>& loss_mask,
                                               std::size_t stride) {
    if (token_stream.size() < 2)
        throw std::invalid_argument("Evaluation token stream must contain at least two tokens");
    if (loss_mask.size() != token_stream.size())
        throw std::invalid_argument("Evaluation loss mask size must match token stream");
    if (stride == 0) throw std::invalid_argument("Evaluation stride must be non-zero");
    for (const auto token : token_stream)
        if (token >= model_.config().vocabulary_size)
            throw std::out_of_range("Evaluation stream contains a token outside the vocabulary");
    for (const auto flag : loss_mask)
        if (flag > 1) throw std::invalid_argument("Evaluation loss mask values must be 0 or 1");

    TrainingStats stats;
    double total_loss = 0.0;
    for (std::size_t target_index = 1; target_index < token_stream.size(); target_index += stride) {
        if (loss_mask[target_index] == 0) continue;
        const std::size_t start = context_start(token_stream, target_index,
                                                model_.config().context_length,
                                                model_.config().vocabulary_size);
        const std::vector<std::size_t> context(
            token_stream.begin() + static_cast<std::ptrdiff_t>(start),
            token_stream.begin() + static_cast<std::ptrdiff_t>(target_index));
        const auto logits = model_.forward_last(context);
        total_loss += cross_entropy_loss(logits, token_stream[target_index]).value();
        ++stats.steps;
    }
    if (stats.steps == 0)
        throw std::invalid_argument("Evaluation loss mask selects no next-token targets");
    stats.average_loss = total_loss / static_cast<double>(stats.steps);
    return stats;
}

void LanguageModelTrainer::set_learning_rate(float learning_rate) {
    optimizer_.set_learning_rate(learning_rate);
}
std::size_t LanguageModelTrainer::steps() const noexcept { return steps_; }

} // namespace flawd
