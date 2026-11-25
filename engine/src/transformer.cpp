#include "flawd/transformer.hpp"

#include <cmath>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <filesystem>
#include <system_error>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace flawd {
namespace {
ModelConfig validated_config(ModelConfig config) {
    config.validate();
    return config;
}

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

void replace_checkpoint_file(const std::filesystem::path& temporary,
                             const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "Unable to atomically replace model checkpoint");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) throw std::system_error(error, "Unable to atomically replace model checkpoint");
#endif
}
}

void ModelConfig::validate() const {
    if (vocabulary_size < 2 || context_length == 0 || embedding_size == 0 ||
        feed_forward_size == 0 || layer_count == 0 || attention_heads == 0 ||
        embedding_size % attention_heads != 0)
        throw std::invalid_argument("Transformer dimensions must be positive and embedding size must be divisible by attention heads");

    const long double width = static_cast<long double>(embedding_size);
    const long double vocabulary = static_cast<long double>(vocabulary_size);
    const long double context = static_cast<long double>(context_length);
    const long double feed_forward = static_cast<long double>(feed_forward_size);
    const long double layers = static_cast<long double>(layer_count);
    const long double estimated_parameters =
        2.0L * vocabulary * width + context * width + vocabulary +
        layers * (4.0L * width * width + 2.0L * width * feed_forward +
                  12.0L * width + feed_forward) + 2.0L * width;
    if (estimated_parameters >
        static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("Transformer dimensions exceed addressable parameter storage");
    constexpr long double maximum_parameters = 5'000'000.0L;
    if (estimated_parameters > maximum_parameters)
        throw std::length_error("Transformer exceeds the 5-million-parameter safety limit");
}

CausalSelfAttention::CausalSelfAttention(std::size_t embedding_size, std::size_t head_count,
                                             std::uint32_t seed)
    : embedding_size_(embedding_size), head_count_(head_count),
      head_size_(head_count == 0 ? 0 : embedding_size / head_count),
      output_(embedding_size, embedding_size, seed + 100) {
    if (embedding_size == 0 || head_count == 0 || embedding_size % head_count != 0)
        throw std::invalid_argument("Attention width must be divisible by a non-zero head count");
    queries_.reserve(head_count_);
    keys_.reserve(head_count_);
    values_.reserve(head_count_);
    for (std::size_t head = 0; head < head_count_; ++head) {
        const auto head_seed = seed + static_cast<std::uint32_t>(head * 3);
        queries_.emplace_back(embedding_size_, head_size_, head_seed);
        keys_.emplace_back(embedding_size_, head_size_, head_seed + 1);
        values_.emplace_back(embedding_size_, head_size_, head_seed + 2);
    }
}

Sequence CausalSelfAttention::forward(const Sequence& input) {
    validate_sequence(input, embedding_size_);
    const std::size_t length = input.size();
    std::vector<Sequence> queries(head_count_), keys(head_count_), values(head_count_);
    for (std::size_t head = 0; head < head_count_; ++head) {
        queries[head].reserve(length);
        keys[head].reserve(length);
        values[head].reserve(length);
        for (const auto& token : input) {
            queries[head].push_back(queries_[head].forward(token));
            keys[head].push_back(keys_[head].forward(token));
            values[head].push_back(values_[head].forward(token));
        }
    }

    Sequence result;
    result.reserve(length);
    const float scale_factor = std::sqrt(static_cast<float>(head_size_));
    for (std::size_t query_index = 0; query_index < length; ++query_index) {
        ScalarVector concatenated;
        concatenated.reserve(embedding_size_);
        for (std::size_t head = 0; head < head_count_; ++head) {
            ScalarVector scores;
            scores.reserve(query_index + 1);
            for (std::size_t key_index = 0; key_index <= query_index; ++key_index) {
                Scalar score(0.0F);
                for (std::size_t dimension = 0; dimension < head_size_; ++dimension)
                    score = score + queries[head][query_index][dimension] * keys[head][key_index][dimension];
                scores.push_back(score / Scalar(scale_factor));
            }
            const auto weights = softmax(scores);
            for (std::size_t dimension = 0; dimension < head_size_; ++dimension) {
                Scalar weighted_sum(0.0F);
                for (std::size_t key_index = 0; key_index <= query_index; ++key_index)
                    weighted_sum = weighted_sum + weights[key_index] * values[head][key_index][dimension];
                concatenated.push_back(weighted_sum);
            }
        }
        result.push_back(output_.forward(concatenated));
    }
    return result;
}

ScalarVector CausalSelfAttention::forward_last(const Sequence& input) {
    validate_sequence(input, embedding_size_);
    ScalarVector concatenated;
    concatenated.reserve(embedding_size_);
    const float scale_factor = std::sqrt(static_cast<float>(head_size_));
    for (std::size_t head = 0; head < head_count_; ++head) {
        const auto query = queries_[head].forward(input.back());
        Sequence keys;
        Sequence values;
        keys.reserve(input.size());
        values.reserve(input.size());
        for (const auto& token : input) {
            keys.push_back(keys_[head].forward(token));
            values.push_back(values_[head].forward(token));
        }
        ScalarVector scores;
        scores.reserve(input.size());
        for (const auto& key : keys) {
            Scalar score(0.0F);
            for (std::size_t dimension = 0; dimension < head_size_; ++dimension)
                score = score + query[dimension] * key[dimension];
            scores.push_back(score / Scalar(scale_factor));
        }
        const auto weights = softmax(scores);
        for (std::size_t dimension = 0; dimension < head_size_; ++dimension) {
            Scalar weighted_sum(0.0F);
            for (std::size_t key_index = 0; key_index < input.size(); ++key_index)
                weighted_sum = weighted_sum + weights[key_index] * values[key_index][dimension];
            concatenated.push_back(weighted_sum);
        }
    }
    return output_.forward(concatenated);
}

std::vector<Scalar> CausalSelfAttention::parameters() {
    std::vector<Scalar> result;
    for (std::size_t head = 0; head < head_count_; ++head) {
        append_parameters(result, queries_[head].parameters());
        append_parameters(result, keys_[head].parameters());
        append_parameters(result, values_[head].parameters());
    }
    append_parameters(result, output_.parameters());
    return result;
}

TransformerBlock::TransformerBlock(std::size_t embedding_size, std::size_t feed_forward_size,
                                   std::size_t attention_heads, std::uint32_t seed)
    : attention_norm_(embedding_size),
      attention_(embedding_size, attention_heads, seed),
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
        const auto pre_activation = feed_forward_in_.forward(normalized_token);
        ScalarVector hidden;
        hidden.reserve(pre_activation.size());
        for (const auto& value : pre_activation) hidden.push_back(flawd::relu(value));
        const auto projected = feed_forward_out_.forward(hidden);
        output.push_back(add_vectors(token, projected));
    }
    return output;
}

ScalarVector TransformerBlock::forward_last(const Sequence& input) {
    if (input.empty()) throw std::invalid_argument("Transformer block input cannot be empty");
    Sequence normalized;
    normalized.reserve(input.size());
    for (const auto& token : input) normalized.push_back(attention_norm_.forward(token));
    const auto attention_output = attention_.forward_last(normalized);
    const auto residual = add_vectors(input.back(), attention_output);
    const auto normalized_token = feed_forward_norm_.forward(residual);
    const auto pre_activation = feed_forward_in_.forward(normalized_token);
    ScalarVector hidden;
    hidden.reserve(pre_activation.size());
    for (const auto& value : pre_activation) hidden.push_back(flawd::relu(value));
    return add_vectors(residual, feed_forward_out_.forward(hidden));
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
    : config_(validated_config(std::move(config))),
      token_embedding_(config_.vocabulary_size, config_.embedding_size, config_.seed),
      position_embedding_(config_.context_length, config_.embedding_size, config_.seed + 100),
      final_norm_(config_.embedding_size),
      language_head_(config_.embedding_size, config_.vocabulary_size, config_.seed + 200) {
    blocks_.reserve(config_.layer_count);
    for (std::size_t i = 0; i < config_.layer_count; ++i)
        blocks_.emplace_back(config_.embedding_size, config_.feed_forward_size,
                             config_.attention_heads,
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

ScalarVector TransformerModel::forward_last(const std::vector<std::size_t>& token_ids) {
    if (token_ids.empty()) throw std::invalid_argument("Model input cannot be empty");
    if (token_ids.size() > config_.context_length)
        throw std::length_error("Input exceeds configured context length");
    Sequence hidden;
    hidden.reserve(token_ids.size());
    for (std::size_t position = 0; position < token_ids.size(); ++position)
        hidden.push_back(add_vectors(token_embedding_.forward(token_ids[position]),
                                     position_embedding_.forward(position)));
    for (std::size_t i = 0; i + 1 < blocks_.size(); ++i)
        hidden = blocks_[i].forward(hidden);
    const auto final_hidden = blocks_.back().forward_last(hidden);
    return language_head_.forward(final_norm_.forward(final_hidden));
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

ModelConfig TransformerModel::read_checkpoint_config(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open checkpoint for reading");
    const auto read = [&](auto& value) {
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input) throw std::runtime_error("Checkpoint header is truncated or unreadable");
    };
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{{'F','L','A','W','D','C','K','P'}};
    if (!input || magic != expected) throw std::runtime_error("Invalid FLAWD checkpoint file");
    std::uint32_t version = 0;
    read(version);
    if (version != 2 && version != 3 && version != 4)
        throw std::runtime_error("Unsupported FLAWD checkpoint version");
    ModelConfig stored;
    read(stored.vocabulary_size);
    read(stored.context_length);
    read(stored.embedding_size);
    read(stored.feed_forward_size);
    read(stored.layer_count);
    read(stored.attention_heads);
    read(stored.seed);
    if (version >= 3) read(stored.tokenizer_fingerprint);
    stored.validate();
    return stored;
}

void TransformerModel::save_checkpoint(const std::string& path, const Adam* optimizer) {
    const std::filesystem::path destination(path);
    std::filesystem::path temporary = destination;
    temporary += ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Unable to open temporary checkpoint for writing");
        const std::array<char, 8> magic{{'F','L','A','W','D','C','K','P'}};
        const std::uint32_t version = 4;
        const auto write = [&](const auto& value) {
            output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        write(version);
        write(config_.vocabulary_size);
        write(config_.context_length);
        write(config_.embedding_size);
        write(config_.feed_forward_size);
        write(config_.layer_count);
        write(config_.attention_heads);
        write(config_.seed);
        write(config_.tokenizer_fingerprint);
        auto model_parameters = parameters();
        const std::uint64_t count = static_cast<std::uint64_t>(model_parameters.size());
        write(count);
        for (const auto& parameter : model_parameters) {
            const float value = parameter.value();
            if (!std::isfinite(value))
                throw std::runtime_error("Cannot save checkpoint with non-finite model weights");
            write(value);
        }
        const std::uint8_t has_optimizer_state = optimizer ? 1 : 0;
        write(has_optimizer_state);
        if (optimizer) {
            const auto state = optimizer->state();
            write(state.learning_rate);
            write(state.beta1);
            write(state.beta2);
            write(state.epsilon);
            write(state.step_count);
            const std::uint64_t moment_count =
                static_cast<std::uint64_t>(state.first_moment.size());
            if (state.first_moment.size() != state.second_moment.size() ||
                state.first_moment.size() != model_parameters.size())
                throw std::runtime_error("Adam state does not match model parameter count");
            write(moment_count);
            for (std::size_t index = 0; index < state.first_moment.size(); ++index) {
                if (!std::isfinite(state.first_moment[index]) ||
                    !std::isfinite(state.second_moment[index]) ||
                    state.second_moment[index] < 0.0F)
                    throw std::runtime_error("Cannot save checkpoint with invalid Adam moments");
                write(state.first_moment[index]);
                write(state.second_moment[index]);
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("Failed while writing model checkpoint");
        output.close();
        if (!output) throw std::runtime_error("Failed while closing model checkpoint");
        replace_checkpoint_file(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void TransformerModel::load_checkpoint(const std::string& path, Adam* optimizer) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open checkpoint for reading");
    const auto read = [&](auto& value) {
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input) throw std::runtime_error("Checkpoint is truncated or unreadable");
    };
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{{'F','L','A','W','D','C','K','P'}};
    if (!input || magic != expected) throw std::runtime_error("Invalid FLAWD checkpoint file");
    std::uint32_t version = 0;
    read(version);
    if (version != 2 && version != 3 && version != 4)
        throw std::runtime_error("Unsupported FLAWD checkpoint version");

    ModelConfig stored;
    read(stored.vocabulary_size);
    read(stored.context_length);
    read(stored.embedding_size);
    read(stored.feed_forward_size);
    read(stored.layer_count);
    read(stored.attention_heads);
    read(stored.seed);
    if (version >= 3) read(stored.tokenizer_fingerprint);
    if (stored.vocabulary_size != config_.vocabulary_size ||
        stored.context_length != config_.context_length ||
        stored.embedding_size != config_.embedding_size ||
        stored.feed_forward_size != config_.feed_forward_size ||
        stored.layer_count != config_.layer_count ||
        stored.attention_heads != config_.attention_heads ||
        stored.tokenizer_fingerprint != config_.tokenizer_fingerprint)
        throw std::runtime_error("Checkpoint model configuration does not match this model");

    std::uint64_t count = 0;
    read(count);
    auto model_parameters = parameters();
    if (count != static_cast<std::uint64_t>(model_parameters.size()))
        throw std::runtime_error("Checkpoint parameter count does not match this model");
    std::vector<float> values;
    values.reserve(model_parameters.size());
    for (std::size_t i = 0; i < model_parameters.size(); ++i) {
        float value = 0.0F;
        read(value);
        if (!std::isfinite(value)) throw std::runtime_error("Checkpoint contains a non-finite weight");
        values.push_back(value);
    }
    AdamState optimizer_state;
    bool has_optimizer_state = false;
    if (version >= 4) {
        std::uint8_t has_state = 0;
        read(has_state);
        if (has_state > 1) throw std::runtime_error("Checkpoint optimizer-state flag is invalid");
        has_optimizer_state = has_state == 1;
        if (has_optimizer_state) {
            read(optimizer_state.learning_rate);
            read(optimizer_state.beta1);
            read(optimizer_state.beta2);
            read(optimizer_state.epsilon);
            read(optimizer_state.step_count);
            if (!(optimizer_state.learning_rate > 0.0F) ||
                !std::isfinite(optimizer_state.learning_rate) ||
                !(optimizer_state.beta1 > 0.0F && optimizer_state.beta1 < 1.0F) ||
                !(optimizer_state.beta2 > 0.0F && optimizer_state.beta2 < 1.0F) ||
                !(optimizer_state.epsilon > 0.0F) || !std::isfinite(optimizer_state.epsilon) ||
                optimizer_state.step_count >
                    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
                throw std::runtime_error("Checkpoint contains invalid Adam hyperparameters");
            std::uint64_t moment_count = 0;
            read(moment_count);
            if (moment_count != count ||
                moment_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
                throw std::runtime_error("Checkpoint Adam moment count does not match model parameters");
            optimizer_state.first_moment.reserve(static_cast<std::size_t>(moment_count));
            optimizer_state.second_moment.reserve(static_cast<std::size_t>(moment_count));
            for (std::size_t index = 0; index < static_cast<std::size_t>(moment_count); ++index) {
                float first = 0.0F;
                float second = 0.0F;
                read(first);
                read(second);
                if (!std::isfinite(first) || !std::isfinite(second) || second < 0.0F)
                    throw std::runtime_error("Checkpoint contains invalid Adam moments");
                optimizer_state.first_moment.push_back(first);
                optimizer_state.second_moment.push_back(second);
            }
        }
    }
    char trailing = 0;
    if (input.read(&trailing, 1))
        throw std::runtime_error("Checkpoint contains unexpected trailing data");
    if (has_optimizer_state && optimizer) optimizer->load_state(optimizer_state);
    for (std::size_t i = 0; i < model_parameters.size(); ++i)
        model_parameters[i].set_value(values[i]);
}


} // namespace flawd
