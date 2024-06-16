//
// Created by julian on 02/03/24.
//

#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

class CudaTextureResource {
public:
    explicit CudaTextureResource(uint texture);

    ~CudaTextureResource();

    void set(const torch::Tensor& tensor);

private:
    cudaGraphicsResource_t resource;
    cudaArray_t array;
    cudaChannelFormatDesc desc;
    cudaExtent ext;
    uint flags;
};
