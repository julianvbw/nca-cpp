//
// Created by julian on 02/03/24.
//

#include <iostream>
#include <glad/gl.h>
#include "cuda_interop.h"

#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cerr << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
        << std::endl; cudaDeviceSynchronize(); } while (0)


//cudaGraphicsResource_t tex_res;
//cudaArray_t tex_arr;
//CUDA_WARN(cudaGraphicsGLRegisterImage(&tex_res, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
//CUDA_WARN(cudaGraphicsMapResources(1, &tex_res));
//CUDA_WARN(cudaGraphicsSubResourceGetMappedArray(&tex_arr, tex_res, 0, 0));
//cudaChannelFormatDesc desc;
//cudaExtent ext;
//uint flags;
//CUDA_WARN(cudaArrayGetInfo(&desc, &ext, &flags, tex_arr));
//void* dev_ptr = nullptr;
//CUDA_WARN(cudaMalloc(&dev_ptr, ext.width*ext.height*4*sizeof(uint8_t)));
//std::cout << ext.width << " " << ext.height << std::endl;
//CUDA_WARN(cudaMemcpy2DFromArray(dev_ptr, ext.width*4*sizeof(uint8_t), tex_arr, 0, 0, ext.width*4*sizeof(uint8_t), ext.height, cudaMemcpyDeviceToDevice));
//{
//auto tex_tensor = torch::from_blob(dev_ptr, { 1, (long long)(ext.height), (long long)(ext.width), 4 }, torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(torch::kCUDA));
//std::cout << tex_tensor.transpose(1, 3) << std::endl;
//
//tex_tensor.slice(2, 8, 12) = 0.0;
//cudaMemcpy2DToArray(tex_arr, 0, 0, tex_tensor.data_ptr<uint8_t>(), ext.width*4*sizeof(uint8_t), ext.width*4*sizeof(uint8_t), ext.height, cudaMemcpyDeviceToDevice);
//}
//CUDA_WARN(cudaGraphicsUnmapResources(1, &tex_res));
//CUDA_WARN(cudaFree(dev_ptr));
//dev_ptr = nullptr;

CudaTextureResource::CudaTextureResource(uint texture) {
    CUDA_WARN(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    CUDA_WARN(cudaGraphicsMapResources(1, &resource));
    CUDA_WARN(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    CUDA_WARN(cudaArrayGetInfo(&desc, &ext, &flags, array));
}

CudaTextureResource::~CudaTextureResource() {
    CUDA_WARN(cudaGraphicsUnmapResources(1, &resource));
}

void CudaTextureResource::set(const torch::Tensor &tensor) {
    cudaMemcpy2DToArray(array, 0, 0, tensor.const_data_ptr<uint8_t>(), ext.width*4*sizeof(uint8_t), ext.width*4*sizeof(uint8_t), ext.height, cudaMemcpyDeviceToDevice);
}
