//
// Created by julian on 03/03/24.
//

#pragma once
#include <../common/utils.h>

namespace trainutils {

    struct Batch {
        torch::Tensor tensor;
        torch::Tensor pool_idx;
    };

    Batch sample_batch(int samplesize, const torch::Tensor& pool, torch::Device dev);

    torch::Tensor batch_argsort(const Batch& batch, const torch::Tensor& img_tensor);

    void batch_seed(Batch& batch, const torch::Tensor& sort_idx, unsigned x, unsigned y);

    void batch_randomdamage(Batch& batch, const torch::Tensor& sort_idx, unsigned damage_count, unsigned damage_radius);

    void update_pool(const Batch& batch, torch::Tensor& pool);
};