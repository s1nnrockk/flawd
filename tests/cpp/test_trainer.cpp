#include "flawd/trainer.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    flawd::ModelConfig config;
    config.vocabulary_size = 259;
    config.context_length = 4;
    config.embedding_size = 4;
    config.feed_forward_size = 8;
    config.layer_count = 1;
    flawd::TransformerModel model(config);
    flawd::LanguageModelTrainer trainer(model, 0.001F);

    const float loss = trainer.train_step({10, 11, 12}, 13);
    if (!std::isfinite(loss) || loss <= 0.0F)
        throw std::runtime_error("Training step returned an invalid loss");
    if (trainer.steps() != 1)
        throw std::runtime_error("Training step counter was not updated");

    const auto stats = trainer.train({10, 11, 12, 13, 14}, 1, 1);
    if (stats.steps != 4 || !std::isfinite(stats.average_loss))
        throw std::runtime_error("Training epoch statistics are invalid");

    const auto validation = trainer.evaluate({10, 11, 12, 13, 14});
    if (validation.steps != 4 || !std::isfinite(validation.average_loss) || validation.average_loss <= 0.0)
        throw std::runtime_error("Validation metrics are invalid");
    if (trainer.steps() != 5)
        throw std::runtime_error("Evaluation unexpectedly changed the optimizer step count");

    const std::vector<std::size_t> masked_tokens{10, 11, 12, 13, 14};
    const std::vector<std::uint8_t> mask{0, 0, 1, 0, 1};
    const auto masked = trainer.train(masked_tokens, mask);
    if (masked.steps != 2)
        throw std::runtime_error("Assistant-only mask did not select the expected targets");
    const auto masked_validation = trainer.evaluate(masked_tokens, mask);
    if (masked_validation.steps != 2 || !std::isfinite(masked_validation.average_loss))
        throw std::runtime_error("Masked validation metrics are invalid");
    if (trainer.steps() != 7)
        throw std::runtime_error("Masked evaluation changed optimizer step count");

    bool rejected_empty_mask = false;
    try {
        trainer.train(masked_tokens, std::vector<std::uint8_t>(masked_tokens.size(), 0));
    } catch (const std::invalid_argument&) {
        rejected_empty_mask = true;
    }
    if (!rejected_empty_mask)
        throw std::runtime_error("Trainer accepted a mask with no supervised targets");

    std::cout << "FLAWD training loop tests passed.\\n";
    return 0;
}
