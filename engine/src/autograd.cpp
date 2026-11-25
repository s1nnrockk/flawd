#include "flawd/autograd.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

namespace flawd {

struct Scalar::Node {
    float data = 0.0F;
    float gradient = 0.0F;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward;
};

Scalar::Scalar(float value) : node_(std::make_shared<Node>()) { node_->data = value; }
Scalar::Scalar(std::shared_ptr<Node> node) : node_(std::move(node)) {}
float Scalar::value() const noexcept { return node_->data; }
float Scalar::grad() const noexcept { return node_->gradient; }
void Scalar::set_value(float value) {
    if (!node_->parents.empty()) throw std::logic_error("Only leaf scalars can be updated");
    node_->data = value;
}
void Scalar::zero_grad() noexcept { node_->gradient = 0.0F; }

void Scalar::backward() const {
    std::vector<std::shared_ptr<Node>> ordered;
    std::vector<std::shared_ptr<Node>> visited;
    std::function<void(const std::shared_ptr<Node>&)> visit = [&](const auto& node) {
        for (const auto& item : visited) if (item == node) return;
        visited.push_back(node);
        for (const auto& parent : node->parents) visit(parent);
        ordered.push_back(node);
    };
    visit(node_);
    node_->gradient += 1.0F;
    for (auto it = ordered.rbegin(); it != ordered.rend(); ++it)
        if ((*it)->backward) (*it)->backward();
}

Scalar operator+(const Scalar& a, const Scalar& b) {
    auto n = std::make_shared<Scalar::Node>();
    n->data = a.value() + b.value();
    n->parents = {a.node_, b.node_};
    const auto x = a.node_, y = b.node_;
    n->backward = [self = n.get(), x, y] { x->gradient += self->gradient; y->gradient += self->gradient; };
    return Scalar(n);
}
Scalar operator-(const Scalar& a, const Scalar& b) {
    auto n = std::make_shared<Scalar::Node>();
    n->data = a.value() - b.value();
    n->parents = {a.node_, b.node_};
    const auto x = a.node_, y = b.node_;
    n->backward = [self = n.get(), x, y] { x->gradient += self->gradient; y->gradient -= self->gradient; };
    return Scalar(n);
}
Scalar operator*(const Scalar& a, const Scalar& b) {
    auto n = std::make_shared<Scalar::Node>();
    n->data = a.value() * b.value();
    n->parents = {a.node_, b.node_};
    const auto x = a.node_, y = b.node_;
    n->backward = [self = n.get(), x, y] {
        x->gradient += y->data * self->gradient;
        y->gradient += x->data * n->gradient;
    };
    return Scalar(n);
}
Scalar operator/(const Scalar& a, const Scalar& b) {
    if (b.value() == 0.0F) throw std::domain_error("Division by zero in autograd");
    auto n = std::make_shared<Scalar::Node>();
    n->data = a.value() / b.value();
    n->parents = {a.node_, b.node_};
    const auto x = a.node_, y = b.node_;
    n->backward = [n, x, y] {
        x->gradient += n->gradient / y->data;
        y->gradient -= n->gradient * x->data / (y->data * y->data);
    };
    return Scalar(n);
}
Scalar tanh(const Scalar& input) {
    auto n = std::make_shared<Scalar::Node>();
    n->data = std::tanh(input.value());
    n->parents = {input.node_};
    const auto x = input.node_;
    n->backward = [self = n.get(), x] { x->gradient += (1.0F - self->data * self->data) * self->gradient; };
    return Scalar(n);
}
Scalar relu(const Scalar& input) {
    auto n = std::make_shared<Scalar::Node>();
    n->data = input.value() > 0.0F ? input.value() : 0.0F;
    n->parents = {input.node_};
    const auto x = input.node_;
    n->backward = [self = n.get(), x] { if (x->data > 0.0F) x->gradient += self->gradient; };
    return Scalar(n);
}
Scalar exp(const Scalar& input) {
    auto n = std::make_shared<Scalar::Node>();
    n->data = std::exp(input.value());
    n->parents = {input.node_};
    const auto x = input.node_;
    n->backward = [self = n.get(), x] { x->gradient += self->data * self->gradient; };
    return Scalar(n);
}

} // namespace flawd
