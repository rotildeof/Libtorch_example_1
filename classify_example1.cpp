#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <torch/torch.h>
#include <vector>

struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(2, 4));
        fc2 = register_module("fc2", torch::nn::Linear(4, 4));
        fc3 = register_module("fc3", torch::nn::Linear(4, 2));
    }
    torch::nn::Linear fc1 = {nullptr};
    torch::nn::Linear fc2 = {nullptr};
    torch::nn::Linear fc3 = {nullptr};

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::softmax(fc3->forward(x), 1);
        return x;
    }
};

int main(int argc, char **argv) {

    std::vector<float> in = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<float> out = {0, 1, 1, 0, 1, 0, 0, 1};
    int n_data = 4;
    auto input = torch::from_blob(in.data(), {n_data, 2});
    auto output = torch::from_blob(out.data(), {n_data, 2});
    std::cout << input << std::endl;
    std::cout << output << std::endl;
    std::shared_ptr<Net> net = std::make_shared<Net>();

    torch::optim::Adam optimizer(net->parameters());

    for (int i = 0; i < 4000; ++i) {
        auto out_ = net->forward(input);
        optimizer.zero_grad();
        auto loss = torch::binary_cross_entropy(out_, output);
        auto loss_value = loss.item<float>();
        loss.backward();
        optimizer.step();
        if (i % 50 == 0) {
            std::cout << "====== LEARNING INFO ====== " << std::endl;
            std::cout << "Loss : " << loss_value << std::endl;
        }
    }
    std::cout << net->forward(input) << std::endl;
    return 0;
}