#include "flawd/trainer.hpp"

#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
template <typename Integer>
Integer read_little_endian(std::istream& input) {
    static_assert(std::is_unsigned<Integer>::value, "Integer must be unsigned");
    Integer value = 0;
    for (std::size_t i = 0; i < sizeof(Integer); ++i) {
        const int byte = input.get();
        if (byte == std::char_traits<char>::eof())
            throw std::runtime_error("Token file is truncated");
        value |= static_cast<Integer>(static_cast<unsigned char>(byte)) << (i * 8);
    }
    return value;
}

struct TokenCorpus {
    std::size_t vocabulary_size = 0;
    std::uint64_t tokenizer_fingerprint = 0;
    std::vector<std::size_t> tokens;
    std::vector<std::uint8_t> loss_mask;
};

std::size_t parse_positive_size(const char* argument, const char* label) {
    const std::string value(argument);
    if (value.empty() || value.front() == '-')
        throw std::invalid_argument(std::string(label) + " must be a positive integer");
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
        throw std::invalid_argument(std::string(label) + " must be a positive integer");
    return static_cast<std::size_t>(parsed);
}

std::size_t parse_epochs(const char* argument) {
    return parse_positive_size(argument, "Epoch count");
}

TokenCorpus load_tokens(const std::string& path, std::size_t expected_vocab) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open token file: " + path);
    char magic[8]{};
    input.read(magic, 8);
    if (!input || std::string(magic, 8) != "FLAWDTOK")
        throw std::runtime_error("Invalid token file magic");
    const auto version = read_little_endian<std::uint32_t>(input);
    const auto vocabulary = read_little_endian<std::uint32_t>(input);
    const auto count = read_little_endian<std::uint64_t>(input);
    if (version != 1 && version != 2 && version != 3) throw std::runtime_error("Unsupported token file version");
    if (vocabulary < 259 || vocabulary > 65536)
        throw std::runtime_error("Token file vocabulary must be between 259 and 65536");
    if (expected_vocab != 0 && vocabulary != expected_vocab)
        throw std::runtime_error("Token file vocabulary does not match model");
    const std::uint64_t tokenizer_fingerprint = version == 3
        ? read_little_endian<std::uint64_t>(input) : 0;
    const std::uint64_t header_size = version == 3 ? 32 : 24;
    if (count < 2) throw std::runtime_error("Token file must contain at least two tokens");
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("Token file is too large for this system");
    constexpr std::uint64_t maximum_tokens_in_memory = 20'000'000;
    if (count > maximum_tokens_in_memory)
        throw std::runtime_error(
            "Token corpus exceeds the 20-million-token in-memory limit; split it into smaller datasets");
    const std::uint64_t bytes_per_token = version >= 2 ? 3 : 2;
    if (count > (std::numeric_limits<std::uint64_t>::max() - header_size) / bytes_per_token)
        throw std::runtime_error("Token file payload size overflows the file format");
    const std::uint64_t expected_size = header_size + count * bytes_per_token;
    input.seekg(0, std::ios::end);
    const auto actual_size = static_cast<std::streamoff>(input.tellg());
    if (actual_size < 0 || static_cast<std::uint64_t>(actual_size) != expected_size)
        throw std::runtime_error("Token file size does not match its header");
    input.seekg(static_cast<std::streamoff>(header_size), std::ios::beg);
    if (!input) throw std::runtime_error("Cannot seek to token payload");

    TokenCorpus corpus;
    corpus.vocabulary_size = vocabulary;
    corpus.tokenizer_fingerprint = tokenizer_fingerprint;
    corpus.tokens.reserve(static_cast<std::size_t>(count));
    corpus.loss_mask.reserve(static_cast<std::size_t>(count));
    std::size_t supervised = 0;
    bool inside_record = false;
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto token = read_little_endian<std::uint16_t>(input);
        std::uint8_t flag = version >= 2 ? read_little_endian<std::uint8_t>(input) : 1;
        if (version == 1 && token == 257) flag = 0;
        if (token >= vocabulary) throw std::runtime_error("Token ID is outside the vocabulary");
        if (flag > 1) throw std::runtime_error("Token loss mask must be 0 or 1");
        if (token == 256)
            throw std::runtime_error("PAD tokens are not valid in an unpadded training corpus");
        if (token == 257) {
            if (inside_record) throw std::runtime_error("Nested BOS marker in token corpus");
            if (flag != 0)
                throw std::runtime_error("BOS tokens cannot be supervised targets");
            inside_record = true;
        } else if (token == 258) {
            if (!inside_record) throw std::runtime_error("EOS marker appears outside a record");
            inside_record = false;
        } else if (!inside_record) {
            throw std::runtime_error("Text token appears outside a BOS/EOS record");
        }
        corpus.tokens.push_back(token);
        corpus.loss_mask.push_back(flag);
        if (i > 0) supervised += flag;
    }
    if (inside_record) throw std::runtime_error("Token corpus ends before the final EOS marker");
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Token file contains unexpected trailing data");
    if (supervised == 0) throw std::runtime_error("Token file contains no supervised targets");
    return corpus;
}
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 11) {
        std::cerr << "Usage: flawd_train <train.tokens> <model.flawd> [epochs] [validation.tokens] [resume.flawd] [context] [embedding] [feed_forward] [layers] [heads]\n";
        return 2;
    }
    try {
        const std::size_t epochs = argc >= 4 ? parse_epochs(argv[3]) : 1;

        auto training = load_tokens(argv[1], 0);
        flawd::ModelConfig config;
        config.vocabulary_size = training.vocabulary_size;
        config.tokenizer_fingerprint = training.tokenizer_fingerprint;
        config.context_length = argc >= 7 ? parse_positive_size(argv[6], "Context length") : 64;
        config.embedding_size = argc >= 8 ? parse_positive_size(argv[7], "Embedding size") : 8;
        config.feed_forward_size = argc >= 9 ? parse_positive_size(argv[8], "Feed-forward size") : 16;
        config.layer_count = argc >= 10 ? parse_positive_size(argv[9], "Layer count") : 1;
        config.attention_heads = argc >= 11 ? parse_positive_size(argv[10], "Attention head count") : 2;
        config.seed = 2026;
        config.validate();


        TokenCorpus validation;
        if (argc >= 5) {
            std::error_code path_error;
            if (std::filesystem::equivalent(argv[1], argv[4], path_error) && !path_error)
                throw std::invalid_argument("Training and validation files must be different");
            validation = load_tokens(argv[4], config.vocabulary_size);
            if (validation.tokenizer_fingerprint != training.tokenizer_fingerprint)
                throw std::invalid_argument("Training and validation tokenizer fingerprints do not match");
            if (training.tokens == validation.tokens &&
                training.loss_mask == validation.loss_mask)
                throw std::invalid_argument("Training and validation corpora contain identical data");
        }

        flawd::TransformerModel model(config);
        flawd::LanguageModelTrainer trainer(model, 0.001F);
        if (argc >= 6 && std::string(argv[5]) != "-") trainer.load_checkpoint(argv[5]);

        constexpr std::size_t early_stopping_patience = 3;
        constexpr double minimum_improvement = 1.0e-4;
        std::size_t stale_epochs = 0;
        std::size_t total_steps = 0;
        double total_loss = 0.0;
        double best_validation_loss = std::numeric_limits<double>::infinity();
        flawd::TrainingStats best_validation_stats;

        if (argc >= 5) {
            best_validation_stats = trainer.evaluate(validation.tokens, validation.loss_mask);
            best_validation_loss = best_validation_stats.average_loss;
            trainer.save_checkpoint(argv[2]);
            std::cout << "Starting validation loss: " << best_validation_loss
                      << " | current checkpoint preserved\\n";
        }

        constexpr std::size_t gradient_accumulation = 4;
        for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
            const auto epoch_stats = trainer.train(training.tokens, training.loss_mask, 1, 1,
                                                   gradient_accumulation);
            total_steps += epoch_stats.steps;
            total_loss += epoch_stats.average_loss * static_cast<double>(epoch_stats.steps);
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " | training loss: " << epoch_stats.average_loss
                      << " | optimizer updates: " << trainer.steps();

            if (argc >= 5) {
                const auto validation_stats = trainer.evaluate(validation.tokens, validation.loss_mask);
                std::cout << " | validation loss: " << validation_stats.average_loss;
                if (validation_stats.average_loss < best_validation_loss - minimum_improvement) {
                    best_validation_loss = validation_stats.average_loss;
                    best_validation_stats = validation_stats;
                    stale_epochs = 0;
                    trainer.save_checkpoint(argv[2]);
                    std::cout << " | best checkpoint saved";
                } else {
                    ++stale_epochs;
                }
                std::cout << '\n';
                if (stale_epochs >= early_stopping_patience) {
                    std::cout << "Early stopping: validation loss did not improve for "
                              << early_stopping_patience << " epochs.\n";
                    break;
                }
            } else {
                std::cout << '\n';
            }
        }

        if (argc < 5) model.save_checkpoint(argv[2]);
        std::cout << "FLAWD training complete\n"
                  << "Steps: " << total_steps << "\n"
                  << "Training loss: " << (total_loss / static_cast<double>(total_steps)) << "\n";
        if (argc >= 5) {
            std::cout << "Best validation steps: " << best_validation_stats.steps << "\n"
                      << "Best validation loss: " << best_validation_loss << "\n";
        }
        std::cout << "Checkpoint: " << argv[2] << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FLAWD training error: " << error.what() << '\n';
        return 1;
    }
}
