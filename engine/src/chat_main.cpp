#include "flawd/transformer.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: flawd_chat <model.flawd>\n";
        return 2;
    }
    try {
        flawd::ModelConfig config;
        config.vocabulary_size = 259;
        config.context_length = 16;
        config.embedding_size = 8;
        config.feed_forward_size = 16;
        config.layer_count = 1;
        config.attention_heads = 2;
        config.seed = 2026;

        flawd::TransformerModel model(config);
        model.load_checkpoint(argv[1]);

        std::cout << "FLAWD local model loaded. Enter a prompt (empty line to quit).\n";
        std::string prompt;
        while (std::cout << "\nYou> " && std::getline(std::cin, prompt)) {
            if (prompt.empty()) break;
            std::vector<std::size_t> tokens;
            tokens.reserve(prompt.size() + 128);
            for (const unsigned char byte : prompt) tokens.push_back(byte);
            std::cout << "FLAWD> " << std::flush;

            for (std::size_t step = 0; step < 128; ++step) {
                const std::size_t start = tokens.size() > config.context_length
                    ? tokens.size() - config.context_length : 0;
                const std::vector<std::size_t> context(
                    tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.end());
                const auto logits = model.forward_last(context);
                const auto best = std::max_element(logits.begin(), logits.end(),
                    [](const flawd::Scalar& left, const flawd::Scalar& right) {
                        return left.value() < right.value();
                    });
                const auto next_token = static_cast<std::size_t>(best - logits.begin());
                if (next_token == 258) break;
                tokens.push_back(next_token);
                if (next_token <= 255) std::cout << static_cast<char>(next_token) << std::flush;
            }
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FLAWD inference error: " << error.what() << '\n';
        return 1;
    }
}
