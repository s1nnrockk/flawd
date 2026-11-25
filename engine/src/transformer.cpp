#include "flawd/transformer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace flawd {
namespace {
void validate_sequence(const Sequence& sequence, std::size_t width) {
    if (sequence.empty()) throw std::invalid_argument("Transformer sequence cannot be empty");
    for (const auto& token : sequence)
        if (token.size() != width) throw std::invalid_argument("Transformer embedding width mismatch");
}
ScalarVector add_vectors(const ScalarVector& left, const ScalarVector& right) {
    if (left.size() != right.size()) throw std::invalid_argument("Residual vector size mismatch");
    ScalarVector result;
    result.reserve(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) result.push_back(left[i] + right[i]);
    return result;
}
void append_parameters(std::vector<Scalar>& destination, std::vector<Scalar> source) {
    destination.insert(destination.end(), source.begin(), source.end());
}
}

void ModelConfig::validate() const {
    if (vocabulary_size < 2 || context_length == 0 || embedding_size == 0 ||
        feed_forward_size == 0 || layer_count == 0)
        throw std::invalid_argument("Transformer configuration dimensions must be positive");
}

CausalSelfAttention::CausalSelfAttention(std::size_t embedding_size, std::uint32_t seed)
    : embedding_size_(embedding_size),
      query_(embedding_size, embedding_size, seed),
      key_(embedding_size, embedding_size, seed + 1),
      value_(embedding_size, embedding_size, seed + 2),
      output_(embedding_size, embedding_size, seed + 3) {
    if (embedding_size == 0) throw std::invalid_argument("Attention embedding size must be non-zero");
}

Sequence CausalSelfAttention::forward(const Sequence& input) {
    validate_sequence(input, embedding_size_);
    const std::size_t length = input.size();
    Sequence queries, keys, values;
    queries.reserve(length); keys.reserve(length); values.reserve(length);
    for (const auto& token : input) {
        queries.push_back(query_.forward(token));
        keys.push_back(key_.forward(token));
        values.push_back(value_.forward(token));
    }

    Sequence result;
    result.reserve(length);
    const float scale_factor = std::sqrt(static_cast<float>(embedding_size_));
    for (std::size_t query_index = 0; query_index < length; ++query_index) {
        ScalarVector scores;
        scores.reserve(query_index + 1);
        for (std::size_t key_index = 0; key_index <= query_index; ++key_index) {
            Scalar score(0.0F);
            for (std::size_t dimension = 0; dimension < embedding_size_; ++dimension)
                score = score + queries[query_index][dimension] * keys[key_index][dimension];
            scores.push_back(score / Scalar(scale_factor));
        }
        const auto weights = softmax(scores);
        ScalarVector context;
        context.reserve(embedding_size_);
        for (std::size_t dimension = 0; dimension < embedding_size_; ++dimension) {
            Scalar weighted_sum(0.0F);
            for (std::size_t key_index = 0; key_index <= query_index; ++key_index)
                weighted_sum = weighted_sum + weights[key_index] * values[key_index][dimension];
            context.push_back(weighted_sum);
        }
        result.push_back(output_.forward(context));
    }
    return result;
}

std::vector<Scalar> CausalSelfAttention::parameters() {
    std::vector<Scalar> result;
    append_parameters(result, query_.parameters());
    append_parameters(result, key_.parameters());
    append_parameters(result, value_.parameters());
    append_parameters(result, output_.parameters());
    return result;
}

TransformerBlock::TransformerBlock(std::size_t embedding_size, std::size_t feed_forward_size,
                                   std::uint32_t seed)
    : attention_norm_(embedding_size),
      attention_(embedding_size, seed),
      feed_forward_norm_(embedding_size),
      feed_forward_in_(embedding_size, feed_forward_size, seed + 10),
      feed_forward_out_(feed_forward_size, embedding_size, seed + 11) {
    if (embedding_size == 0 || feed_forward_size == 0)
        throw std::invalid_argument("Transformer block dimensions must be non-zero");
}

Sequence TransformerBlock::forward(const Sequence& input) {
    if (input.empty()) throw std::invalid_argument("Transformer block input cannot be empty");
    Sequence normalized;
    normalized.reserve(input.size());
    for (const auto& token : input) normalized.push_back(attention_norm_.forward(token));
    const auto attention_output = attention_.forward(normalized);

    Sequence residual;
    residual.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        residual.push_back(add_vectors(input[i], attention_output[i]));

    Sequence output;
    output.reserve(residual.size());
    for (const auto& token : residual) {
        const auto normalized_token = feed_forward_norm_.forward(token);
        const auto hidden = flawd::relu(feed_forward_in_.forward(normalized_token));
        const auto projected = feed_forward_out_.forward(hidden);
        output.push_back(add_vectors(token, projected));
    }
    return output;
}

std::vector<Scalar> TransformerBlock::parameters() {
    std::vector<Scalar> result;
    append_parameters(result, attention_norm_.parameters());
    append_parameters(result, attention_.parameters());
    append_parameters(result, feed_forward_norm_.parameters());
    append_parameters(result, feed_forward_in_.parameters());
    append_parameters(result, feed_forward_out_.parameters());
    return result;
}

TransformerModel::TransformerModel(ModelConfig config)
    : config_(std::move(config)),
      token_embedding_(config_.vocabulary_size, config_.embedding_size, config_.seed),
      position_embedding_(config_.context_length, config_.embedding_size, config_.seed + 100),
      final_norm_(config_.embedding_size),
      language_head_(config_.embedding_size, config_.vocabulary_size, config_.seed + 200) {
    config_.validate();
    blocks_.reserve(config_.layer_count);
    for (std::size_t i = 0; i < config_.layer_count; ++i)
        blocks_.emplace_back(config_.embedding_size, config_.feed_forward_size,
                             config_.seed + static_cast<std::uint32_t>(1000 + i * 20));
}

Sequence TransformerModel::forward(const std::vector<std::size_t>& token_ids) {
    if (token_ids.empty()) throw std::invalid_argument("Model input cannot be empty");
    if (token_ids.size() > config_.context_length)
        throw std::length_error("Input exceeds configured context length");

    Sequence hidden;
    hidden.reserve(token_ids.size());
    for (std::size_t position = 0; position < token_ids.size(); ++position) {
        const auto token = token_embedding_.forward(token_ids[position]);
        const auto positional = position_embedding_.forward(position);
        hidden.push_back(add_vectors(token, positional));
    }
    for (auto& block : blocks_) hidden = block.forward(hidden);

    Sequence logits;
    logits.reserve(hidden.size());
    for (const auto& token : hidden)
        logits.push_back(language_head_.forward(final_norm_.forward(token)));
    return logits;
}

std::vector<Scalar> TransformerModel::parameters() {
    std::vector<Scalar> result;
    append_parameters(result, token_embedding_.parameters());
    append_parameters(result, position_embedding_.parameters());
    for (auto& block : blocks_) append_parameters(result, block.parameters());
    append_parameters(result, final_norm_.parameters());
    append_parameters(result, language_head_.parameters());
    return result;
}
const ModelConfig& TransformerModel::config() const noexcept { return config_; }

} // namespace flawd
