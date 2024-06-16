//
// Created by julian on 25/02/24.
//

#pragma once
#include <torch/torch.h>
#include <lodepng.h>
#include <random>

namespace utils {
    torch::Tensor load_image_to_tensor(const char *filename, unsigned &width, unsigned &height);

    unsigned int save_tensor_as_image(const torch::Tensor &tensor, const char *filename);

    torch::Tensor
    circle_mask(unsigned x, unsigned y, unsigned radius, unsigned width, unsigned height, torch::Device dev);

    void seed(at::Tensor tensor, unsigned channel_dim, torch::ArrayRef<unsigned> seedpos, torch::ArrayRef<unsigned> dims);
}