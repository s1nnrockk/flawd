#include "flawd/tensor.hpp"

#include <iostream>

int main() {
    const auto a = flawd::Tensor::from_values({1.0F, 2.0F, 3.0F});
    const auto b = flawd::Tensor::from_values({4.0F, 5.0F, 6.0F});
    const auto result = flawd::add(a, b);

    std::cout << "FLAWD Engine foundation ready\n";
    std::cout << "Tensor addition: ";
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << result[i];
    }
    std::cout << '\n';

    const auto probabilities = flawd::softmax(flawd::Tensor::from_values({1.0F, 2.0F, 3.0F}));
    std::cout << "Softmax: ";
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << probabilities[i];
    }
    std::cout << '\n';
    return 0;
}
