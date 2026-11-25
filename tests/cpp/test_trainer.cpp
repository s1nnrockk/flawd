#include "flawd/trainer.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <cstdio>
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

    bool rejected_invalid_mask = false;
    try {
        trainer.evaluate(masked_tokens, std::vector<std::uint8_t>{0, 2, 0, 0, 0});
    } catch (const std::invalid_argument&) {
        rejected_invalid_mask = true;
    }
    if (!rejected_invalid_mask)
        throw std::runtime_error("Trainer accepted an invalid loss mask value");

    flawd::TransformerModel accumulated_model(config);
    flawd::LanguageModelTrainer accumulated_trainer(accumulated_model);
    const auto accumulated_stats = accumulated_trainer.train(
        std::vector<std::size_t>{10, 11, 12, 13, 14}, 1, 1, 3);
    if (accumulated_stats.steps != 4 || !std::isfinite(accumulated_stats.average_loss))
        throw std::runtime_error("Gradient accumulation changed target accounting");
    if (accumulated_trainer.steps() != 2)
        throw std::runtime_error("Gradient accumulation did not produce the expected two optimizer updates");

    flawd::TransformerModel masked_accumulated_model(config);
    flawd::LanguageModelTrainer masked_accumulated_trainer(masked_accumulated_model);
    const auto masked_accumulated = masked_accumulated_trainer.train(
        masked_tokens, mask, 1, 1, 4);
    if (masked_accumulated.steps != 2 || masked_accumulated_trainer.steps() != 1)
        throw std::runtime_error("Masked gradient accumulation mishandled a partial final batch");

    bool rejected_zero_accumulation = false;
    try { accumulated_trainer.train(masked_tokens, 1, 1, 0); }
    catch (const std::invalid_argument&) { rejected_zero_accumulation = true; }
    if (!rejected_zero_accumulation)
        throw std::runtime_error("Trainer accepted zero gradient accumulation");

    const std::string resume_path = "flawd_training_resume_test.bin";
    flawd::TransformerModel uninterrupted_model(config);
    flawd::LanguageModelTrainer uninterrupted_trainer(uninterrupted_model, 0.001F);
    const std::vector<std::size_t> resume_tokens{257, 10, 11, 12, 13, 258};
    const std::vector<std::uint8_t> resume_mask{0, 0, 1, 1, 1, 1};
    uninterrupted_trainer.train(resume_tokens, resume_mask, 1, 1, 2);
    uninterrupted_trainer.save_checkpoint(resume_path);

    flawd::TransformerModel resumed_model(config);
    flawd::LanguageModelTrainer resumed_trainer(resumed_model, 0.001F);
    resumed_trainer.load_checkpoint(resume_path);
    if (resumed_trainer.steps() != uninterrupted_trainer.steps())
        throw std::runtime_error("Training checkpoint did not restore Adam update count");

    const std::vector<std::size_t> continuation_tokens{257, 10, 11, 12, 13, 14, 258};
    const std::vector<std::uint8_t> continuation_mask{0, 0, 1, 1, 1, 1, 1};
    uninterrupted_trainer.train(continuation_tokens, continuation_mask, 1, 1, 2);
    resumed_trainer.train(continuation_tokens, continuation_mask, 1, 1, 2);
    const auto uninterrupted_parameters = uninterrupted_model.parameters();
    const auto resumed_parameters = resumed_model.parameters();
    if (uninterrupted_parameters.size() != resumed_parameters.size())
        throw std::runtime_error("Resumed model parameter count changed");
    for (std::size_t index = 0; index < uninterrupted_parameters.size(); ++index) {
        if (std::fabs(uninterrupted_parameters[index].value() -
                      resumed_parameters[index].value()) > 1e-7F)
            throw std::runtime_error("Resumed Adam training diverged from uninterrupted training");
    }
    std::remove(resume_path.c_str());
    std::remove((resume_path + ".tmp").c_str());

    flawd::TransformerModel non_finite_model(config);
    flawd::LanguageModelTrainer non_finite_trainer(non_finite_model);
    auto non_finite_parameters = non_finite_model.parameters();
    non_finite_parameters.back().set_value(std::numeric_limits<float>::quiet_NaN());
    bool rejected_non_finite_gradient = false;
    try { non_finite_trainer.train_step({257, 10}, 11); }
    catch (const std::runtime_error&) { rejected_non_finite_gradient = true; }
    if (!rejected_non_finite_gradient || non_finite_trainer.steps() != 0)
        throw std::runtime_error("Adam did not reject non-finite training state safely");

    flawd::TransformerModel boundary_model_a(config);
    flawd::TransformerModel boundary_model_b(config);
    flawd::LanguageModelTrainer boundary_trainer_a(boundary_model_a);
    flawd::LanguageModelTrainer boundary_trainer_b(boundary_model_b);
    const std::vector<std::uint8_t> boundary_mask{0, 0, 0, 0, 0, 1, 0};
    const auto boundary_a = boundary_trainer_a.evaluate(
        std::vector<std::size_t>{257, 65, 258, 257, 66, 67, 258}, boundary_mask);
    const auto boundary_b = boundary_trainer_b.evaluate(
        std::vector<std::size_t>{257, 90, 258, 257, 66, 67, 258}, boundary_mask);
    if (std::abs(boundary_a.average_loss - boundary_b.average_loss) > 1.0e-6)
        throw std::runtime_error("Training context leaked across BOS document boundaries");

    std::cout << "FLAWD training loop tests passed.\n";
    return 0;
}
