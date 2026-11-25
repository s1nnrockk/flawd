#include "flawd/transformer.hpp"
#include "flawd/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t sample_next_token(const flawd::ScalarVector& logits, float temperature,
                              std::size_t top_k, std::mt19937& random) {
    if (logits.empty()) throw std::runtime_error("Model returned an empty vocabulary distribution");
    for (const auto& logit : logits)
        if (!std::isfinite(logit.value())) throw std::runtime_error("Model returned non-finite logits");
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
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: flawd_chat <model.flawd> [temperature] [top_k] [tokenizer.bpe]\n";
        return 2;
    }
    try {
        const float temperature = argc >= 3 ? std::stof(argv[2]) : 0.8F;
        const std::size_t top_k = argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 40;
        if (!std::isfinite(temperature) || temperature < 0.0F)
            throw std::invalid_argument("Temperature must be a finite non-negative number");
        if (top_k == 0) throw std::invalid_argument("top_k must be positive");
        std::mt19937 random(2026);
        const auto config = flawd::TransformerModel::read_checkpoint_config(argv[1]);
        std::unique_ptr<flawd::ByteBPETokenizer> bpe_tokenizer;
        if (argc == 5) {
            bpe_tokenizer = std::make_unique<flawd::ByteBPETokenizer>(argv[4]);
            if (bpe_tokenizer->vocabulary_size() != config.vocabulary_size)
                throw std::invalid_argument("BPE tokenizer vocabulary does not match checkpoint");
            if (bpe_tokenizer->fingerprint() != config.tokenizer_fingerprint)
                throw std::invalid_argument("BPE tokenizer file does not match the checkpoint tokenizer");
        } else if (config.vocabulary_size != 259 || config.tokenizer_fingerprint != 0) {
            throw std::invalid_argument("This checkpoint requires its matching .bpe tokenizer file");
        }
        flawd::TransformerModel model(config);
        model.load_checkpoint(argv[1]);

        std::cout << "FLAWD local model loaded. Enter a prompt (empty line to quit).\n";
        std::vector<std::size_t> conversation_tokens{257};
        const auto append_text = [&](const std::string& text) {
            if (bpe_tokenizer) {
                const auto encoded = bpe_tokenizer->encode(text);
                conversation_tokens.insert(conversation_tokens.end(), encoded.begin(), encoded.end());
            } else {
                for (const unsigned char byte : text) conversation_tokens.push_back(byte);
            }
        };

        std::string prompt;
        while (std::cout << "\nYou> " && std::getline(std::cin, prompt)) {
            if (prompt.empty()) break;
            append_text("<|user|>\n" + prompt + "\n<|assistant|>\n");
            std::cout << "FLAWD> " << std::flush;
            std::string generated_text;

            for (std::size_t step = 0; step < 128; ++step) {
                const std::size_t start = conversation_tokens.size() > config.context_length
                    ? conversation_tokens.size() - config.context_length : 0;
                const std::vector<std::size_t> context(
                    conversation_tokens.begin() + static_cast<std::ptrdiff_t>(start),
                    conversation_tokens.end());
                const auto logits = model.forward_last(context);
                const auto next_token = sample_next_token(logits, temperature, top_k, random);
                if (next_token == 258 || next_token == 257) break;
                if (next_token == 256) continue;
                conversation_tokens.push_back(next_token);

                std::string piece;
                if (next_token <= 255) {
                    piece.push_back(static_cast<char>(next_token));
                } else if (bpe_tokenizer && next_token >= 259) {
                    piece = bpe_tokenizer->decode(next_token);
                }
                std::cout << piece << std::flush;
                generated_text += piece;
            }
            if (generated_text.empty() || generated_text.back() != '\n')
                append_text("\n");
            std::cout << '\n';
            const std::size_t history_limit =
                config.context_length > std::numeric_limits<std::size_t>::max() / 8
                    ? std::numeric_limits<std::size_t>::max()
                    : config.context_length * 8;
            if (conversation_tokens.size() > history_limit) {
                const auto excess = conversation_tokens.size() - history_limit;
                conversation_tokens.erase(
                    conversation_tokens.begin(),
                    conversation_tokens.begin() + static_cast<std::ptrdiff_t>(excess));
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FLAWD inference error: " << error.what() << '\n';
        return 1;
    }
}
