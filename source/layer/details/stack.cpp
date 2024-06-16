// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-12-9.
#include "stack.hpp"
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"



namespace kuiper_infer {
    using namespace arma;
StackLayer::StackLayer(int startdim)
    : NonParamLayer("Stack"), dim(startdim) {}

StatusCode StackLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the Stack layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the Stack layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the Stack "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  
    
  
    int start_dim = dim;


  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) 
  {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);

    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input tensor array in the Stack layer has"
                    " an empty tensor "
                 << i << " th";
      return StatusCode::kInferInputsEmpty;
    }

    arma::Cube<float>data=input->data();
    arma::Cube<float> Stack_result;
    std::shared_ptr<Tensor<float>> output ;


    outputs.at(i) = input;
   
  }
  return StatusCode::kSuccess;
}

StatusCode StackLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& Stack_layer) {
  if (!op) {
    LOG(ERROR) << "The Stack operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the Stack layer is empty.";
    return StatusCode::kParseParameterError;
  }

//   if (!op->has_parameter("min"))  {
//     LOG(ERROR) << "Can not find the min value parameter in torch.Stack";
//     return StatusCode::kParseParameterError;
//   }

    if (!op->has_parameter("dim"))  {
        LOG(ERROR) << "Can not find the max value parameter in torch.Stack";
        return StatusCode::kParseParameterError;
    }


  auto target_dim = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim"));
    // auto target_max = std::dynamic_pointer_cast<RuntimeParameterFloat>(params.at("max"));
 

  if (target_dim == nullptr)  {
    LOG(ERROR) << "The  parameter in the Stack layer is empty.";
    return StatusCode::kParseParameterError;
  }





  Stack_layer = std::make_shared<StackLayer>(target_dim->value);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kStackCreateInstance(StackLayer::CreateInstance, "torch.stack");

}  // namespace kuiper_infer