#include "flawd/transformer.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    flawd::ModelConfig config;
    config.vocabulary_size = 259;
    config.context_length = 8;
    config.embedding_size = 8;
    config.feed_forward_size = 16;
    config.layer_count = 2;
    config.attention_heads = 2;
    flawd::TransformerModel model(config);

    const auto first = model.forward({10, 20});
    const auto changed_future = model.forward({10, 21});
    const auto last_logits = model.forward_last({10, 20});
    if (last_logits.size() != config.vocabulary_size)
        throw std::runtime_error("forward_last output shape mismatch");
    for (std::size_t i = 0; i < config.vocabulary_size; ++i) {
        if (std::fabs(last_logits[i].value() - first.back()[i].value()) > 1e-5F)
            throw std::runtime_error("forward_last differs from the final full-sequence logits");
    }
    if (first.size() != 2 || first[0].size() != config.vocabulary_size)
        throw std::runtime_error("Transformer output shape mismatch");
    for (std::size_t i = 0; i < config.vocabulary_size; ++i) {
        if (std::fabs(first[0][i].value() - changed_future[0][i].value()) > 1e-5F)
            throw std::runtime_error("Causal attention leaked a future token");
    }
    const auto original_weight = model.parameters().front().value();
    const std::string checkpoint_path = "flawd_test_checkpoint.bin";
    model.save_checkpoint(checkpoint_path);
    auto mutable_parameters = model.parameters();
    mutable_parameters.front().set_value(original_weight + 1.0F);
    model.load_checkpoint(checkpoint_path);
    if (std::fabs(model.parameters().front().value() - original_weight) > 1e-6F)
        throw std::runtime_error("Checkpoint did not restore model weights");
    std::remove(checkpoint_path.c_str());

    if (model.parameters().empty())
        throw std::runtime_error("Transformer model has no trainable parameters");

    bool rejected = false;
    try { (void)model.forward({1,2,3,4,5,6,7,8,9}); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Model accepted input beyond context length");

    std::cout << "FLAWD Transformer tests passed.\n";
    return 0;
}
