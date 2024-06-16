//
// Created by julian on 25/02/24.
//

#include "utils.h"

namespace utils {

    torch::Tensor load_image_to_tensor(const char *filename, unsigned &width, unsigned &height) {
        std::vector<unsigned char> image;
        unsigned error = lodepng::decode(image, width, height, filename);
        std::vector<float> fimg(image.size());
        std::copy(image.begin(), image.end(), fimg.begin());
        return torch::from_blob(fimg.data(), {width, height, 4}).divide(255.0f).transpose(0, 2);
    }

    unsigned int save_tensor_as_image(const torch::Tensor &tensor, const char *filename) {
        auto sz = tensor.sizes();
        auto img = tensor.transpose(0, 2).clamp(0.0, 1.0).multiply(255.0f).to(torch::kU8);
        if (!img.is_contiguous()) {
            img = img.contiguous();
        }
        return lodepng::encode(filename, (unsigned char *) img.data_ptr(), sz[1], sz[2]);
    }

    torch::Tensor circle_mask(unsigned x, unsigned y, unsigned r, unsigned w, unsigned h, torch::Device dev) {
        auto mgrid = torch::meshgrid({torch::arange(0, (int)w, dev), torch::arange(0, (int)h, dev)}, "ij");
        auto circ = (mgrid[0] - (int) r).square() + (mgrid[1] - (int) r).square();
        return circ.greater((int) (r * r)).roll({x - r, y - r}, {0, 1});
    }

    void seed(at::Tensor tensor, unsigned chn, torch::ArrayRef<unsigned> seedpos, torch::ArrayRef<unsigned> dims){
        tensor.zero_().slice(chn, 3)
            .slice(dims[0], seedpos[0], seedpos[0] + 1)
            .slice(dims[1], seedpos[1], seedpos[1] + 1) = 1.0f;
    }
}