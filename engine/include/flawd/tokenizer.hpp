#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flawd {

class ByteBPETokenizer {
public:
    explicit ByteBPETokenizer(const std::string& path);
    std::size_t vocabulary_size() const noexcept;
    std::vector<std::size_t> encode(const std::string& text) const;
    std::string decode(std::size_t token_id) const;

private:
    std::size_t vocabulary_size_ = 259;
    std::vector<std::pair<std::size_t, std::size_t>> merges_;
    std::unordered_map<std::uint32_t, std::size_t> ranks_;
};

} // namespace flawd
