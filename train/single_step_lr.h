//
// Created by julian on 11/02/24.
//
#pragma once

#include <torch/optim/schedulers/lr_scheduler.h>

namespace torch {
    namespace optim {

        class TORCH_API SingleStepLR : public LRScheduler {
        public:

            SingleStepLR(torch::optim::Optimizer& optimizer,
                   const unsigned step,
                   const double gamma = 0.1);

        private:
            std::vector<double> get_lrs() override;

            const unsigned step_;
            const double gamma_;

        };
    } // namespace optim
} // namespace torch
