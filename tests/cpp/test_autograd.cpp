#include "flawd/nn.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void near(float a, float b, const char* message) { expect(std::fabs(a-b)<1e-4F,message); }
}

int main() {
    flawd::Scalar x(3.0F);
    const auto y = x * x + flawd::Scalar(2.0F) * x;
    y.backward();
    near(y.value(), 15.0F, "scalar expression value failed");
    near(x.grad(), 8.0F, "scalar gradient failed");

    flawd::Scalar positive(4.0F);
    const auto root = flawd::sqrt(positive);
    root.backward();
    near(root.value(), 2.0F, "sqrt value failed");
    near(positive.grad(), 0.25F, "sqrt gradient failed");

    const flawd::ScalarVector logits{flawd::Scalar(1.0F), flawd::Scalar(2.0F), flawd::Scalar(3.0F)};
    const auto classification_loss = flawd::cross_entropy_loss(logits, 2);
    expect(classification_loss.value() > 0.0F, "cross entropy loss must be positive");
    classification_loss.backward();
    expect(logits[2].grad() < 0.0F, "cross entropy target gradient must be negative");

    const auto extreme_loss = flawd::cross_entropy_loss(
        {flawd::Scalar(1000.0F), flawd::Scalar(-1000.0F)}, 1);
    near(extreme_loss.value(), 2000.0F, "stable cross entropy failed for extreme logits");

    flawd::LayerNorm normalization(3);
    const auto normalized = normalization.forward({flawd::Scalar(1.0F), flawd::Scalar(2.0F), flawd::Scalar(3.0F)});
    const float normalized_sum = normalized[0].value() + normalized[1].value() + normalized[2].value();
    near(normalized_sum, 0.0F, "layer normalization mean failed");
    const auto class_probabilities = flawd::softmax({flawd::Scalar(1.0F), flawd::Scalar(2.0F), flawd::Scalar(3.0F)});
    near(class_probabilities[0].value() + class_probabilities[1].value() + class_probabilities[2].value(),
         1.0F, "vector softmax normalization failed");

    flawd::Embedding embedding(16, 4, 9);
    const auto embedded = embedding.forward(3);
    expect(embedded.size() == 4, "embedding dimension mismatch");
    expect(embedding.parameters().size() == 64, "embedding parameter count mismatch");
    bool bad_token = false;
    try { (void)embedding.forward(16); } catch (const std::out_of_range&) { bad_token = true; }
    expect(bad_token, "embedding accepted an out-of-range token");

    flawd::Linear linear(1, 1, 7);
    auto params = linear.parameters();
    flawd::SGD optimizer(params, 0.05F);
    flawd::Sequential model;
    model.add(linear);
    const flawd::ScalarVector input{flawd::Scalar(2.0F)};
    const flawd::ScalarVector target{flawd::Scalar(1.0F)};
    float first_loss = 0.0F;
    for (int step = 0; step < 80; ++step) {
        optimizer.zero_grad();
        const auto prediction = model.forward(input);
        const auto loss = flawd::mean_squared_error(prediction, target);
        if (step == 0) first_loss = loss.value();
        loss.backward();
        optimizer.step();
    }
    const auto final_loss = flawd::mean_squared_error(model.forward(input), target).value();
    expect(final_loss < first_loss, "SGD training did not reduce loss");
    flawd::Linear adam_linear(1, 1, 13);
    flawd::Sequential adam_model;
    adam_model.add(adam_linear);
    flawd::Adam adam(adam_model.parameters(), 0.02F);
    const float adam_initial = flawd::mean_squared_error(adam_model.forward(input), target).value();
    for (int step = 0; step < 100; ++step) {
        adam.zero_grad();
        const auto loss = flawd::mean_squared_error(adam_model.forward(input), target);
        loss.backward();
        adam.step();
    }
    const float adam_final = flawd::mean_squared_error(adam_model.forward(input), target).value();
    expect(adam_final < adam_initial, "Adam training did not reduce loss");

    std::cout << "All FLAWD autograd and neural-layer tests passed.\n";
}
