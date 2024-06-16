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
#include "sum.hpp"
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"



namespace kuiper_infer {
    using namespace arma;
SumLayer::SumLayer(int startdim)
    : NonParamLayer("Sum"), dim(startdim) {}

StatusCode SumLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the Sum layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the Sum layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the Sum "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  
    
  
    int start_dim = dim;


  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) 
  {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);

    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input tensor array in the Sum layer has"
                    " an empty tensor "
                 << i << " th";
      return StatusCode::kInferInputsEmpty;
    }

    arma::Cube<float>data=input->data();
    arma::Cube<float> sum_result;
    std::shared_ptr<Tensor<float>> output ;

    arma::Cube<float> *out;

            switch(dim) {
                case 0: // Sum along the 1st dimension
                    sum_result = arma::sum(data, 2);

                    
                    break;

                case 1: // Sum along the 2nd dimension
                    sum_result = arma::sum(data, 0);
                    break;
                case 2: // Sum along the 3rd dimension
                    sum_result = arma::sum(data, 1);
                    break;
                default:
                    std::cerr << "Invalid dimension specified for summation" << std::endl;
                    return StatusCode::kInferParameterError;
                    
            }
            out=&sum_result;
            outputs.at(i) = std::make_shared<Tensor<float>>(out,sum_result.n_slices  ,sum_result.n_rows,sum_result.n_cols);
   
  }
  return StatusCode::kSuccess;
}

StatusCode SumLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& Sum_layer) {
  if (!op) {
    LOG(ERROR) << "The Sum operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the Sum layer is empty.";
    return StatusCode::kParseParameterError;
  }

//   if (!op->has_parameter("min"))  {
//     LOG(ERROR) << "Can not find the min value parameter in torch.Sum";
//     return StatusCode::kParseParameterError;
//   }

    if (!op->has_parameter("dim"))  {
        LOG(ERROR) << "Can not find the max value parameter in torch.Sum";
        return StatusCode::kParseParameterError;
    }


  auto target_dim = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("dim"));
    // auto target_max = std::dynamic_pointer_cast<RuntimeParameterFloat>(params.at("max"));
 

  if (target_dim == nullptr)  {
    LOG(ERROR) << "The min or max parameter in the Sum layer is empty.";
    return StatusCode::kParseParameterError;
  }





  Sum_layer = std::make_shared<SumLayer>(target_dim->value[0]);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSumCreateInstance(SumLayer::CreateInstance, "torch.sum");

}  // namespace kuiper_infer