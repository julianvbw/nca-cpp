//
// Created by julian on 02/02/24.
//

#include "model.h"

#define TINT(X) X.item<int>()

NCAModelImpl::NCAModelImpl(int channels) :
    channels(register_buffer("channels",
            torch::tensor(channels))
    ),
    update(register_module("update",
    torch::nn::Sequential(
            torch::nn::Conv2d(3*channels, 128, 1),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(128, channels, 1)
    ))),
    perception(register_module("perception",
    torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, 3*channels, 3).groups(channels).bias(false)
    )))
{
    init_update();
    init_perception();
}

void NCAModelImpl::init_perception(){
    const float sobel_blob[9] = {
            -1, +0, +1,
            -2, +0, +2,
            -1, +0, +1
    };
    const float eye_blob[9] = {
            0, 0, 0,
            0, 1, 0,
            0, 0, 0
    };

    const auto sobel_tensor = torch::from_blob((void*) &sobel_blob[0], {3, 3})
            .unsqueeze(0)
            .repeat({TINT(channels), 1, 1, 1});
    const auto eye_tensor = torch::from_blob((void*) &eye_blob[0], {3, 3})
            .unsqueeze(0)
            .repeat({TINT(channels), 1, 1, 1});

    torch::NoGradGuard guard;
    perception->weight.set_(torch::cat({sobel_tensor, sobel_tensor.transpose(2, 3), eye_tensor}));
}

void NCAModelImpl::init_update() {
    torch::NoGradGuard guard;
    update->named_parameters()["2.weight"].zero_();
    update->named_parameters()["2.bias"].zero_();
}

torch::Tensor NCAModelImpl::stochastic_update(const torch::Tensor& grid, const torch::Tensor& updated) {
    const auto sz = grid.sizes();
    const auto mask = torch::rand({sz[0], sz[2], sz[3]})
            .less(0.5)
            .unsqueeze(1)
            .repeat({1, sz[1], 1, 1})
            .to(grid.dtype())
            .to(grid.device());
    return grid + updated * mask;
}

torch::Tensor NCAModelImpl::alive_masking(const torch::Tensor& grid) {
    const auto sz = grid.sizes();
    const auto padded = torch::pad(grid, {1, 1, 1, 1}, "circular");
    const auto alive = torch::nn::functional::max_pool2d(padded.slice(1, 3, 4), torch::nn::MaxPool2dOptions(3).stride(1))
            .greater(0.1f)
            .repeat({1, sz[1], 1, 1})
            .to(grid.dtype());
    return grid * alive;
}

torch::Tensor NCAModelImpl::forward(const torch::Tensor& grid) {
    const auto padded = torch::pad(grid, {1, 1, 1, 1}, "circular");
    const auto perceived = perception->forward(padded);
    auto updated = stochastic_update(grid, update->forward(perceived));
    updated = alive_masking(updated);
    return updated;
}

void NCAModelImpl::normalize_grad_() {
    for (auto& param : update->named_parameters()){
        param->mutable_grad() /= param->grad().norm() + 1e-8;
    }
}
