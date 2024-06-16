//
// Created by julian on 03/03/24.
//

#include "utils.h"

namespace trainutils {
    Batch sample_batch(int samplesize, const torch::Tensor& pool, torch::Device dev){
        auto idx = torch::multinomial(torch::ones({pool.size(0)}, dev), samplesize);
        return {pool.index_select(0, idx), idx};
    }

    torch::Tensor batch_argsort(const Batch& batch, const torch::Tensor& img_tensor){
        const auto loss = torch::mean(
                torch::mse_loss(batch.tensor.slice(1, 0, 4), img_tensor.repeat({batch.tensor.size(0), 1, 1, 1}), torch::Reduction::None),
                {1, 2, 3});
        return torch::argsort(loss);
    }

    void batch_seed(Batch& batch, const torch::Tensor& sort_idx, unsigned x, unsigned y){
        const auto& worst_idx = sort_idx[batch.tensor.size(0)-1].item<int>();
        utils::seed(batch.tensor.slice(0, worst_idx, worst_idx+1), 1, {x, y}, {2, 3});
    }

    void batch_randomdamage(Batch& batch, const torch::Tensor& sort_idx, unsigned damage_count, unsigned damage_radius){
        const auto& top3_idx = sort_idx.slice(0, batch.tensor.size(0)-1 - damage_count);
        auto damage_targets = batch.tensor.index_select(0, top3_idx);

        int w = batch.tensor.size(2), h = batch.tensor.size(3);
        auto rndx = torch::randint(w, {damage_count});
        auto rndy = torch::randint(h, {damage_count});

        for (int i = 0; i < damage_count; ++i) {
            auto mask = utils::circle_mask(rndx[i].item<int>(), rndy[i].item<int>(), damage_radius, w, h, batch.tensor.device());
            damage_targets[i].mul_(mask);
        }
        batch.tensor.index_copy(0, top3_idx, damage_targets);
    }

    void update_pool(const Batch& batch, torch::Tensor& pool){
        pool.index_copy_(0, batch.pool_idx, batch.tensor);
    }
};