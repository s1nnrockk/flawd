#pragma once

#include "flawd/transformer.hpp"

#include <cstddef>
#include <vector>

namespace flawd {

struct TrainingStats {
    std::size_t steps = 0;
    double average_loss = 0.0;
};

class LanguageModelTrainer {
public:
    explicit LanguageModelTrainer(TransformerModel& model, float learning_rate = 0.001F);
    float train_step(const std::vector<std::size_t>& input_tokens, std::size_t target_token);
    TrainingStats train(const std::vector<std::size_t>& token_stream,
                        std::size_t epochs = 1, std::size_t stride = 1);
    TrainingStats evaluate(const std::vector<std::size_t>& token_stream,
                           std::size_t stride = 1);
    void set_learning_rate(float learning_rate);
    std::size_t steps() const noexcept;

private:
    TransformerModel& model_;
    Adam optimizer_;
    std::size_t steps_ = 0;
};

} // namespace flawd
