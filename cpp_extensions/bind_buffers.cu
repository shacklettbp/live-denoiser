#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "bind_buffers.h"

#define CU_CHECK_THROW(x)                                                                    \
  do {                                                                                       \
    CUresult result = x;                                                                     \
    if (result != CUDA_SUCCESS) {                                                            \
      const char *msg;                                                                       \
      cuGetErrorName(result, &msg);                                                          \
      throw std::runtime_error(std::string("CUDA Error: " #x " failed with error ") + msg);  \
    }                                                                                        \
  } while(0)

#define CUDA_CHECK_THROW(x)                                                                                          \
  do {                                                                                                               \
    cudaError_t result = x;                                                                                          \
    if (result != cudaSuccess)                                                                                       \
      throw std::runtime_error(std::string("CUDA Error: " #x " failed with error ") + cudaGetErrorString(result));   \
  } while(0)

std::pair<void *, void *> getDevicePtr(void *handle, uint64_t bytes)
{
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = handle;
    externalMemoryHandleDesc.size = bytes;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t externalMemory;
    CUDA_CHECK_THROW(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

    cudaExternalMemoryBufferDesc bufferDesc;
    bufferDesc.offset = 0;
    bufferDesc.size = externalMemoryHandleDesc.size;
    bufferDesc.flags = 0;

    void *dev_ptr;
    CUDA_CHECK_THROW(cudaExternalMemoryGetMappedBuffer(&dev_ptr, externalMemory, &bufferDesc));

    return std::make_pair(dev_ptr, externalMemory);
}
