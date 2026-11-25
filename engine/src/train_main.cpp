#include "flawd/trainer.hpp"

#include <cstdint>
#include <fstream>
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
    std::vector<std::size_t> tokens;
    std::vector<std::uint8_t> loss_mask;
};

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
    if (version != 1 && version != 2) throw std::runtime_error("Unsupported token file version");
    if (vocabulary != expected_vocab) throw std::runtime_error("Token file vocabulary does not match model");
    if (count < 2) throw std::runtime_error("Token file must contain at least two tokens");
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("Token file is too large for this system");
    const std::uint64_t bytes_per_token = version == 2 ? 3 : 2;
    if (count > (std::numeric_limits<std::uint64_t>::max() - 24) / bytes_per_token)
        throw std::runtime_error("Token file payload size overflows the file format");
    const std::uint64_t expected_size = 24 + count * bytes_per_token;
    input.seekg(0, std::ios::end);
    const auto actual_size = input.tellg();
    if (actual_size < 0 || static_cast<std::uint64_t>(actual_size) != expected_size)
        throw std::runtime_error("Token file size does not match its header");
    input.seekg(24, std::ios::beg);
    if (!input) throw std::runtime_error("Cannot seek to token payload");

    TokenCorpus corpus;
    corpus.tokens.reserve(static_cast<std::size_t>(count));
    corpus.loss_mask.reserve(static_cast<std::size_t>(count));
    std::size_t supervised = 0;
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto token = read_little_endian<std::uint16_t>(input);
        const auto flag = version == 2 ? read_little_endian<std::uint8_t>(input) : 1;
        if (token >= vocabulary) throw std::runtime_error("Token ID is outside the vocabulary");
        if (flag > 1) throw std::runtime_error("Token loss mask must be 0 or 1");
        corpus.tokens.push_back(token);
        corpus.loss_mask.push_back(flag);
        supervised += flag;
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Token file contains unexpected trailing data");
    if (supervised == 0) throw std::runtime_error("Token file contains no supervised targets");
    return corpus;
}
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: flawd_train <train.tokens> <model.flawd> [epochs] [validation.tokens] [resume.flawd]\n";
        return 2;
    }
    try {
        const std::size_t epochs = argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 1;
        if (epochs == 0) throw std::invalid_argument("Epoch count must be positive");

        flawd::ModelConfig config;
        config.vocabulary_size = 259;
        config.context_length = 64;
        config.embedding_size = 8;
        config.feed_forward_size = 16;
        config.layer_count = 1;
        config.attention_heads = 2;
        config.seed = 2026;

        auto training = load_tokens(argv[1], config.vocabulary_size);
        TokenCorpus validation;
        if (argc >= 5) validation = load_tokens(argv[4], config.vocabulary_size);

        flawd::TransformerModel model(config);
        if (argc == 6) model.load_checkpoint(argv[5]);
        flawd::LanguageModelTrainer trainer(model, 0.001F);

        constexpr std::size_t early_stopping_patience = 3;
        constexpr double minimum_improvement = 1.0e-4;
        std::size_t stale_epochs = 0;
        std::size_t total_steps = 0;
        double total_loss = 0.0;
        double best_validation_loss = std::numeric_limits<double>::infinity();
        flawd::TrainingStats best_validation_stats;

        for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
            const auto epoch_stats = trainer.train(training.tokens, training.loss_mask, 1);
            total_steps += epoch_stats.steps;
            total_loss += epoch_stats.average_loss * static_cast<double>(epoch_stats.steps);
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " | training loss: " << epoch_stats.average_loss;

            if (argc >= 5) {
                const auto validation_stats = trainer.evaluate(validation.tokens, validation.loss_mask);
                std::cout << " | validation loss: " << validation_stats.average_loss;
                if (validation_stats.average_loss < best_validation_loss - minimum_improvement) {
                    best_validation_loss = validation_stats.average_loss;
                    best_validation_stats = validation_stats;
                    stale_epochs = 0;
                    model.save_checkpoint(argv[2]);
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
