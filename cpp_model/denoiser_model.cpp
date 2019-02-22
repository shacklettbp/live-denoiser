#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include <vector>
#include <utility>

#include <iostream>
#include "denoiser_model.h"

struct DenoiserBackward : public torch::autograd::Function
{
    std::shared_ptr<DenoiserModelImpl> impl;

    explicit DenoiserBackward(const std::shared_ptr<DenoiserModelImpl> &impl_ptr) : impl(impl_ptr) {}

    torch::autograd::variable_list apply(torch::autograd::variable_list &&grads) override
    {
        auto &grad_output = grads[0];
        auto &grad_ei = grads[1];
        auto grad_result = impl->backward(grad_output, grad_ei);

        torch::autograd::variable_list grad_inputs;
        for (auto &grad : grad_result) {
            grad_inputs.emplace_back(grad);
        }
        return grad_inputs;
    }
};

struct DenoiserModel : public torch::nn::Module
{
    std::shared_ptr<DenoiserModelImpl> impl;

    DenoiserModel() : impl(std::make_shared<DenoiserModelImpl>(
                [this](const std::string &name, at::Tensor &param) 
                {
                    param = this->register_parameter(name, param);
                }))
    {
    }

    at::Tensor forward(at::Tensor full_input)
    {
        std::shared_ptr<torch::autograd::Function> grad_fn;
        if (torch::autograd::GradMode::is_enabled()) {
            torch::autograd::variable_list inputs {full_input};
            for (auto &param : parameters()) {
                inputs.push_back(param);
            }
            grad_fn = std::make_shared<DenoiserBackward>(impl);
            grad_fn->set_next_edges(torch::autograd::collect_next_edges(inputs));
        }

        auto out = impl->forward(full_input);

        torch::autograd::variable_list outputs {out.first, out.second};

        torch::autograd::set_history(outputs, grad_fn);

        return outputs;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    torch::python::bind_module<DenoiserModel>(m, "DenoiserModel").
        def(py::init<>()).
        def("forward", &DenoiserModel::forward);
}
