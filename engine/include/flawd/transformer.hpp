#pragma once

#include "flawd/nn.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flawd {

using Sequence = std::vector<ScalarVector>;

struct ModelConfig {
    std::size_t vocabulary_size = 259;
    std::size_t context_length = 32;
    std::size_t embedding_size = 32;
    std::size_t feed_forward_size = 64;
    std::size_t layer_count = 2;
    std::uint32_t seed = 2026;

    void validate() const;
};

class CausalSelfAttention {
public:
    CausalSelfAttention(std::size_t embedding_size, std::uint32_t seed = 1);
    Sequence forward(const Sequence& input);
    std::vector<Scalar> parameters();

private:
    std::size_t embedding_size_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear output_;
};

class TransformerBlock {
public:
    TransformerBlock(std::size_t embedding_size, std::size_t feed_forward_size,
                     std::uint32_t seed = 1);
    Sequence forward(const Sequence& input);
    std::vector<Scalar> parameters();

private:
    LayerNorm attention_norm_;
    CausalSelfAttention attention_;
    LayerNorm feed_forward_norm_;
    Linear feed_forward_in_;
    Linear feed_forward_out_;
};

class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config);
    Sequence forward(const std::vector<std::size_t>& token_ids);
    std::vector<Scalar> parameters();
    const ModelConfig& config() const noexcept;

private:
    ModelConfig config_;
    Embedding token_embedding_;
    Embedding position_embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm final_norm_;
    Linear language_head_;
};

} // namespace flawd
