#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <utility>
#include <vector>

#include "denoiser_model.h"

ConvolutionData::ConvolutionData(const std::string &name,
                                 LayerInitCB cb,
                                 int in_channels, int out_channels,
                                 int kernel_w, int kernel_h,
                                 int padding_w, int padding_h,
                                 bool activation)
    : layer_name(name),
      input(),
      output(),
      kernel(),
      conv(),
      algo(),
      workspace(nullptr),
      conv_weights(at::zeros({out_channels, kernel_w, kernel_h})),
      conv_bias(at::zeros({out_channels}))
{
    cb(layer_name + "_weights", conv_weights);
    cb(layer_name + "_bias", conv_bias);
}

DenoiserModelImpl::DenoiserModelImpl(LayerInitCB &cb)
    : handle(at::native::getCudnnHandle()),
      layers {
        { "encoder_1_1", cb, 9, 48 },
        { "encoder_2_1", cb, 48, 48 },
        { "encoder_3_1", cb, 48, 48 },
        { "encoder_4_1", cb, 48, 48 },
        { "encoder_5_1", cb, 48, 48 },
        { "bottleneck_1_1", cb, 48, 48 },
        { "decoder_1_1", cb, 8+48, 96 },
        { "decoder_1_2", cb, 6, 96 },
        { "decoder_2_1", cb, 6+48, 96 },
        { "decoder_2_2", cb, 6, 96 },
        { "decoder_3_1", cb, 6+48, 96 },
        { "decoder_3_2", cb, 6, 96 },
        { "decoder_4_1", cb, 6+48, 96 },
        { "decoder_4_2", cb, 6, 96 },
        { "decoder_5_1", cb, 6+9, 64 },
        { "decoder_5_2", cb, 4, 32 },
        { "final", cb, 32, 3, 3, 3, 1, 1, false }
    }
{}

std::pair<at::Tensor, at::Tensor> DenoiserModelImpl::forward(at::Tensor full_input)
{
    return std::make_pair(full_input, full_input);
}
std::vector<at::Tensor> DenoiserModelImpl::backward(at::Tensor grad_output, at::Tensor grad_ei)
{
    return {};
}
