//
// Created by julian on 02/02/24.
//
#pragma once
#include <torch/nn.h>

class NCAModelImpl: public torch::nn::Module {
public /* methods */:
    explicit NCAModelImpl(int channels = 16);

    torch::Tensor forward(const torch::Tensor& grid);

    void normalize_grad_();

public /* variables */:
    torch::Tensor channels;
    torch::nn::Sequential update;

private /* methods */:
    void init_perception();

    void init_update();

    static torch::Tensor stochastic_update(const torch::Tensor& grid, const torch::Tensor& updated);

    static torch::Tensor alive_masking(const torch::Tensor& grid);

private /* variables */:
    torch::nn::Conv2d perception;
};

TORCH_MODULE(NCAModel);