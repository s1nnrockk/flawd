#include "flawd/tokenizer.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace flawd {
namespace {
std::uint32_t read_u32(std::istream& input) {
    std::uint32_t value = 0;
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        const int byte = input.get();
        if (byte == std::char_traits<char>::eof())
            throw std::runtime_error("BPE tokenizer header is truncated");
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return value;
}
std::uint16_t read_u16(std::istream& input) {
    const int low = input.get();
    const int high = input.get();
    if (low == std::char_traits<char>::eof() || high == std::char_traits<char>::eof())
        throw std::runtime_error("BPE tokenizer merge table is truncated");
    return static_cast<std::uint16_t>(
        static_cast<unsigned char>(low) |
        (static_cast<std::uint16_t>(static_cast<unsigned char>(high)) << 8));
}
std::uint32_t pair_key(std::size_t left, std::size_t right) {
    return (static_cast<std::uint32_t>(left) << 16) |
           static_cast<std::uint32_t>(right);
}
}

ByteBPETokenizer::ByteBPETokenizer(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open BPE tokenizer: " + path);
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{{'F','L','A','W','D','B','P','E'}};
    if (!input || magic != expected)
        throw std::runtime_error("Invalid FLAWD BPE tokenizer file");
    const auto version = read_u32(input);
    vocabulary_size_ = read_u32(input);
    if (version != 1) throw std::runtime_error("Unsupported BPE tokenizer version");
    if (vocabulary_size_ < 259 || vocabulary_size_ > 65536)
        throw std::runtime_error("BPE tokenizer vocabulary size is invalid");

    const std::size_t merge_count = vocabulary_size_ - 259;
    input.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::streamoff>(input.tellg());
    const std::uint64_t expected_size = 16ULL + static_cast<std::uint64_t>(merge_count) * 4ULL;
    if (file_size < 0 || static_cast<std::uint64_t>(file_size) != expected_size)
        throw std::runtime_error("BPE tokenizer file size does not match its header");
    input.seekg(16, std::ios::beg);
    if (!input) throw std::runtime_error("Unable to seek to BPE merge table");

    merges_.reserve(merge_count);
    ranks_.reserve(merge_count);
    for (std::size_t index = 0; index < merge_count; ++index) {
        const std::size_t left = read_u16(input);
        const std::size_t right = read_u16(input);
        const std::size_t new_token = 259 + index;
        if (left >= new_token || right >= new_token ||
            left == 256 || left == 257 || left == 258 ||
            right == 256 || right == 257 || right == 258)
            throw std::runtime_error("BPE tokenizer contains an invalid merge");
        const auto key = pair_key(left, right);
        if (!ranks_.emplace(key, index).second)
            throw std::runtime_error("BPE tokenizer contains a duplicate merge");
        merges_.emplace_back(left, right);
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("BPE tokenizer contains trailing data");
}

std::size_t ByteBPETokenizer::vocabulary_size() const noexcept {
    return vocabulary_size_;
}

std::vector<std::size_t> ByteBPETokenizer::encode(const std::string& text) const {
    std::vector<std::size_t> tokens;
    tokens.reserve(text.size());
    for (const unsigned char byte : text) tokens.push_back(byte);

    while (tokens.size() > 1) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::size_t best_left = 0;
        std::size_t best_right = 0;
        for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
            const auto found = ranks_.find(pair_key(tokens[index], tokens[index + 1]));
            if (found != ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_left = tokens[index];
                best_right = tokens[index + 1];
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) break;

        std::vector<std::size_t> merged;
        merged.reserve(tokens.size());
        for (std::size_t index = 0; index < tokens.size();) {
            if (index + 1 < tokens.size() &&
                tokens[index] == best_left && tokens[index + 1] == best_right) {
                merged.push_back(259 + best_rank);
                index += 2;
            } else {
                merged.push_back(tokens[index]);
                ++index;
            }
        }
        tokens = std::move(merged);
    }
    return tokens;
}

std::string ByteBPETokenizer::decode(std::size_t token_id) const {
    if (token_id >= vocabulary_size_)
        throw std::out_of_range("BPE token ID is outside the vocabulary");
    if (token_id >= 256 && token_id <= 258)
        return {};
    std::string output;
    std::vector<std::size_t> pending{token_id};
    while (!pending.empty()) {
        const auto current = pending.back();
        pending.pop_back();
        if (current <= 255) {
            output.push_back(static_cast<char>(current));
        } else {
            const auto index = current - 259;
            if (index >= merges_.size())
                throw std::runtime_error("BPE token expansion is invalid");
            pending.push_back(merges_[index].second);
            pending.push_back(merges_[index].first);
        }
    }
    return output;
}

} // namespace flawd
