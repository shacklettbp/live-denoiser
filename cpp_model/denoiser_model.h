#include <ATen/ATen.h>
#include <cudnn.h>
#include <utility>
#include <functional>

using LayerInitCB = std::function<void(const std::string &, at::Tensor)>;

struct ConvolutionData
{
    std::string layer_name;
    cudnnTensorDescriptor_t input;
    cudnnTensorDescriptor_t output;
    cudnnFilterDescriptor_t kernel;
    cudnnConvolutionDescriptor_t conv;
    cudnnConvolutionFwdAlgo_t algo;
    void *workspace;

    at::Tensor conv_weights;
    at::Tensor conv_bias;

    ConvolutionData(const std::string &name,
                    LayerInitCB cb,
                    int in_channels, int out_channels,
                    int kernel_w=3, int kernel_h=3,
                    int padding_w=1, int padding_h=1, bool activation=true);
};

struct DenoiserModelImpl
{
    std::vector<ConvolutionData> layers;

    DenoiserModelImpl(LayerInitCB &cb);

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor full_input);
    std::vector<at::Tensor> backward(at::Tensor grad_output, at::Tensor grad_ei);
};
