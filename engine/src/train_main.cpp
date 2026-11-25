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

std::vector<std::size_t> load_tokens(const std::string& path, std::size_t expected_vocab) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open token file: " + path);
    char magic[8]{};
    input.read(magic, 8);
    if (!input || std::string(magic, 8) != "FLAWDTOK")
        throw std::runtime_error("Invalid token file magic");
    const auto version = read_little_endian<std::uint32_t>(input);
    const auto vocabulary = read_little_endian<std::uint32_t>(input);
    const auto count = read_little_endian<std::uint64_t>(input);
    if (version != 1) throw std::runtime_error("Unsupported token file version");
    if (vocabulary != expected_vocab) throw std::runtime_error("Token file vocabulary does not match model");
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("Token file is too large for this system");

    std::vector<std::size_t> tokens;
    tokens.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto token = read_little_endian<std::uint16_t>(input);
        if (token >= vocabulary) throw std::runtime_error("Token ID is outside the vocabulary");
        tokens.push_back(token);
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Token file contains unexpected trailing data");
    return tokens;
}
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: flawd_train <train.tokens> <model.flawd> [epochs] [validation.tokens]\n";
        return 2;
    }
    try {
        const std::size_t epochs = argc == 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 1;
        if (epochs == 0) throw std::invalid_argument("Epoch count must be positive");

        flawd::ModelConfig config;
        config.vocabulary_size = 259;
        config.context_length = 16;
        config.embedding_size = 8;
        config.feed_forward_size = 16;
        config.layer_count = 1;
        config.attention_heads = 2;
        config.seed = 2026;

        auto tokens = load_tokens(argv[1], config.vocabulary_size);
        flawd::TransformerModel model(config);
        flawd::LanguageModelTrainer trainer(model, 0.001F);
        const auto stats = trainer.train(tokens, epochs);
        model.save_checkpoint(argv[2]);

        std::cout << "FLAWD training complete\n"
                  << "Steps: " << stats.steps << "\n"
                  << "Training loss: " << stats.average_loss << "\n";
        if (argc == 5) {
            const auto validation_tokens = load_tokens(argv[4], config.vocabulary_size);
            const auto validation = trainer.evaluate(validation_tokens);
            std::cout << "Validation steps: " << validation.steps << "\n"
                      << "Validation loss: " << validation.average_loss << "\n";
        }
        std::cout << "Checkpoint: " << argv[2] << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FLAWD training error: " << error.what() << '\n';
        return 1;
    }
}
