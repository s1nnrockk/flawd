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
    std::cout << "All FLAWD autograd and neural-layer tests passed.\n";
}
