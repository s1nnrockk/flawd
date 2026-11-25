#include "flawd/trainer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

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

    std::cout << "FLAWD training loop tests passed.\n";
    return 0;
}
