#include <torch/extension.h>
#include <vector>
#include <utility>
#include "bind_buffers.h"

std::pair<std::vector<at::Tensor>, at::Tensor> bind_buffers(const std::vector<std::pair<void *, uint64_t>> &input_ptrs, std::pair<void *, uint64_t> output_ptr, const std::vector<int64_t> &sizes_raw)
{
    torch::IntList sizes(sizes_raw);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    std::vector<at::Tensor> inputs;
    for (auto pair : input_ptrs)
    {
        auto input_ptr_pair = getDevicePtr(pair.first, pair.second);

        inputs.emplace_back(std::move(torch::from_blob(input_ptr_pair.first, sizes, options)));
    }

    auto output_ptr_pair = getDevicePtr(output_ptr.first, output_ptr.second);

    auto output = torch::from_blob(output_ptr_pair.first, sizes, options);

    return std::make_pair(inputs, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bind_buffers", &bind_buffers, "Bind DX handles to cuda at::Tensor");
}
