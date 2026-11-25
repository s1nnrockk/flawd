#include "flawd/transformer.hpp"

#include <cmath>
#include <filesystem>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main() {
    flawd::ModelConfig config;
    config.vocabulary_size = 263;
    config.tokenizer_fingerprint = 0x0123456789abcdefULL;
    config.context_length = 8;
    config.embedding_size = 8;
    config.feed_forward_size = 16;
    config.layer_count = 2;
    config.attention_heads = 2;
    flawd::TransformerModel model(config);

    auto oversized_config = config;
    oversized_config.embedding_size = std::numeric_limits<std::size_t>::max();
    oversized_config.attention_heads = 1;
    bool rejected_overflowing_dimensions = false;
    try { oversized_config.validate(); }
    catch (const std::length_error&) { rejected_overflowing_dimensions = true; }
    if (!rejected_overflowing_dimensions)
        throw std::runtime_error("Model config accepted dimensions that overflow parameter storage");
    bool rejected_before_allocation = false;
    try { flawd::TransformerModel invalid_model(oversized_config); }
    catch (const std::length_error&) { rejected_before_allocation = true; }
    if (!rejected_before_allocation)
        throw std::runtime_error("Transformer constructor did not validate dimensions before allocation");

    auto excessive_config = config;
    excessive_config.embedding_size = 4096;
    excessive_config.feed_forward_size = 4096;
    excessive_config.attention_heads = 1;
    bool rejected_excessive_parameters = false;
    try { flawd::TransformerModel invalid_model(excessive_config); }
    catch (const std::length_error&) { rejected_excessive_parameters = true; }
    if (!rejected_excessive_parameters)
        throw std::runtime_error("Transformer accepted a model above the parameter safety limit");

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
    const std::vector<std::size_t> longer_input{10, 20, 30, 40};
    const auto longer_full = model.forward(longer_input);
    const auto longer_last = model.forward_last(longer_input);
    for (std::size_t i = 0; i < config.vocabulary_size; ++i) {
        if (std::fabs(longer_last[i].value() - longer_full.back()[i].value()) > 1e-5F)
            throw std::runtime_error("Optimized final-token path differs on a longer sequence");
    }
    const auto original_weight = model.parameters().front().value();
    const std::string checkpoint_path = "flawd_test_checkpoint.bin";
    model.save_checkpoint(checkpoint_path);
    const auto checkpoint_config = flawd::TransformerModel::read_checkpoint_config(checkpoint_path);
    if (checkpoint_config.vocabulary_size != config.vocabulary_size ||
        checkpoint_config.context_length != config.context_length ||
        checkpoint_config.embedding_size != config.embedding_size ||
        checkpoint_config.feed_forward_size != config.feed_forward_size ||
        checkpoint_config.layer_count != config.layer_count ||
        checkpoint_config.attention_heads != config.attention_heads ||
        checkpoint_config.tokenizer_fingerprint != config.tokenizer_fingerprint)
        throw std::runtime_error("Checkpoint configuration reader returned incorrect dimensions");
    auto mutable_parameters = model.parameters();
    mutable_parameters.front().set_value(original_weight + 1.0F);
    model.save_checkpoint(checkpoint_path);
    model.parameters().front().set_value(original_weight);
    model.load_checkpoint(checkpoint_path);
    if (std::fabs(model.parameters().front().value() - (original_weight + 1.0F)) > 1e-6F)
        throw std::runtime_error("Checkpoint overwrite did not atomically replace model weights");
    if (std::filesystem::exists(checkpoint_path + ".tmp"))
        throw std::runtime_error("Checkpoint save left a temporary file behind");
    auto non_finite_parameters = model.parameters();
    non_finite_parameters.front().set_value(std::numeric_limits<float>::quiet_NaN());
    bool rejected_non_finite_save = false;
    try { model.save_checkpoint(checkpoint_path); }
    catch (const std::runtime_error&) { rejected_non_finite_save = true; }
    if (!rejected_non_finite_save)
        throw std::runtime_error("Checkpoint writer accepted a non-finite weight");
    non_finite_parameters.front().set_value(0.0F);
    model.load_checkpoint(checkpoint_path);
    if (std::fabs(model.parameters().front().value() - (original_weight + 1.0F)) > 1e-6F)
        throw std::runtime_error("Failed checkpoint write damaged the previous valid checkpoint");
    if (std::filesystem::exists(checkpoint_path + ".tmp"))
        throw std::runtime_error("Failed checkpoint write left a temporary file behind");
    auto different_heads = config;
    different_heads.attention_heads = 4;
    flawd::TransformerModel incompatible_model(different_heads);
    bool rejected_checkpoint = false;
    try { incompatible_model.load_checkpoint(checkpoint_path); }
    catch (const std::runtime_error&) { rejected_checkpoint = true; }
    if (!rejected_checkpoint)
        throw std::runtime_error("Checkpoint accepted incompatible attention configuration");
    auto different_tokenizer = config;
    ++different_tokenizer.tokenizer_fingerprint;
    flawd::TransformerModel mismatched_tokenizer_model(different_tokenizer);
    bool rejected_tokenizer = false;
    try { mismatched_tokenizer_model.load_checkpoint(checkpoint_path); }
    catch (const std::runtime_error&) { rejected_tokenizer = true; }
    if (!rejected_tokenizer)
        throw std::runtime_error("Checkpoint accepted a mismatched tokenizer fingerprint");
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
