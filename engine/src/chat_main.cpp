#include "flawd/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t sample_next_token(const flawd::ScalarVector& logits, float temperature,
                              std::size_t top_k, std::mt19937& random) {
    if (logits.empty()) throw std::runtime_error("Model returned an empty vocabulary distribution");
    if (temperature <= 0.0F) {
        const auto best = std::max_element(logits.begin(), logits.end(),
            [](const flawd::Scalar& left, const flawd::Scalar& right) {
                return left.value() < right.value();
            });
        return static_cast<std::size_t>(best - logits.begin());
    }

    std::vector<std::size_t> candidates(logits.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) candidates[i] = i;
    std::partial_sort(candidates.begin(), candidates.begin() + std::min(top_k, candidates.size()),
                      candidates.end(), [&](std::size_t left, std::size_t right) {
        return logits[left].value() > logits[right].value();
    });
    candidates.resize(std::min(top_k, candidates.size()));
    const float maximum = logits[candidates.front()].value();
    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (const auto index : candidates)
        weights.push_back(std::exp((logits[index].value() - maximum) / temperature));
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(random)];
}
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: flawd_chat <model.flawd> [temperature] [top_k]\n";
        return 2;
    }
    try {
        const float temperature = argc >= 3 ? std::stof(argv[2]) : 0.8F;
        const std::size_t top_k = argc == 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 40;
        if (!std::isfinite(temperature) || temperature < 0.0F)
            throw std::invalid_argument("Temperature must be a finite non-negative number");
        if (top_k == 0) throw std::invalid_argument("top_k must be positive");
        std::mt19937 random(2026);
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
            const std::string formatted_prompt = "<|user|>\n" + prompt + "\n<|assistant|>\n";
            for (const unsigned char byte : formatted_prompt) tokens.push_back(byte);
            std::cout << "FLAWD> " << std::flush;

            for (std::size_t step = 0; step < 128; ++step) {
                const std::size_t start = tokens.size() > config.context_length
                    ? tokens.size() - config.context_length : 0;
                const std::vector<std::size_t> context(
                    tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.end());
                const auto logits = model.forward_last(context);
                const auto next_token = sample_next_token(logits, temperature, top_k, random);
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
