#include "flawd/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void write_u32(std::ostream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8)
        output.put(static_cast<char>((value >> shift) & 0xffU));
}
void write_u16(std::ostream& output, std::uint16_t value) {
    output.put(static_cast<char>(value & 0xffU));
    output.put(static_cast<char>((value >> 8) & 0xffU));
}
}

int main() {
    const std::string path = "flawd_test_tokenizer.bpe";
    {
        std::ofstream output(path, std::ios::binary);
        const char magic[8] = {'F','L','A','W','D','B','P','E'};
        output.write(magic, 8);
        write_u32(output, 1);
        write_u32(output, 263);
        write_u16(output, 104); write_u16(output, 101);  // he
        write_u16(output, 259); write_u16(output, 108);  // hel
        write_u16(output, 260); write_u16(output, 108);  // hell
        write_u16(output, 261); write_u16(output, 111);  // hello
    }

    flawd::ByteBPETokenizer tokenizer(path);
    if (tokenizer.vocabulary_size() != 263)
        throw std::runtime_error("BPE vocabulary size was read incorrectly");
    std::uint64_t expected_fingerprint = 14695981039346656037ULL;
    {
        std::ifstream input(path, std::ios::binary);
        char byte = 0;
        while (input.get(byte))
            expected_fingerprint = (expected_fingerprint ^
                static_cast<std::uint8_t>(byte)) * 1099511628211ULL;
    }
    if (tokenizer.fingerprint() != expected_fingerprint)
        throw std::runtime_error("Native BPE tokenizer fingerprint is incorrect");
    const auto encoded = tokenizer.encode("hello hello");
    if (encoded != std::vector<std::size_t>{262, 32, 262})
        throw std::runtime_error("C++ BPE encoder does not match the merge table");
    if (tokenizer.decode(262) != "hello")
        throw std::runtime_error("C++ BPE decoder failed to expand a merged token");

    bool rejected_out_of_range = false;
    try { (void)tokenizer.decode(263); }
    catch (const std::out_of_range&) { rejected_out_of_range = true; }
    if (!rejected_out_of_range)
        throw std::runtime_error("BPE decoder accepted an out-of-range token");

    std::remove(path.c_str());
    std::cout << "FLAWD BPE tokenizer tests passed.\n";
    return 0;
}
