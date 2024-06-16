//
// Created by julian on 11/02/24.
//

#include "single_step_lr.h"

namespace torch::optim {

    SingleStepLR::SingleStepLR(torch::optim::Optimizer& optimizer,
                   const unsigned step,
                   const double gamma) :
            LRScheduler(optimizer),
            step_(step),
            gamma_(gamma) {}

    std::vector<double> SingleStepLR::get_lrs() {
        if(step_count_ < step_)
            return get_current_lrs();
        else {
            std::vector<double> lrs = get_current_lrs();
            std::transform(lrs.begin(), lrs.end(), lrs.begin(),
                           [this](const double& v){ return this->gamma_ * v; });
            return lrs;
        }
    }

} // namespace torch::optim
