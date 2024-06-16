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
#include "unbind.hpp"
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"



namespace kuiper_infer {
    using namespace arma;
UnbindLayer::UnbindLayer(uint32_t targetdim)
    : NonParamLayer("Unbind"), target_dim(targetdim) {}

StatusCode UnbindLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the unbind layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the unbind layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the unbind "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  int32_t dimension = target_dim;
    
  int32_t total_dims = 4;  // NCHW



  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) 
  {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);

    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input tensor array in the unbind layer has"
                    " an empty tensor "
                 << i << " th";
      return StatusCode::kInferInputsEmpty;
    }
    // const uint32_t input_h = input->rows();
    // const uint32_t input_w = input->cols();
    // const uint32_t input_c = input->channels();
    // cout<<input_h<<input_w<<input_c<<"!!!!!!"<<endl;
        
    // arma::Mat<float> subdata;
    // arma::Cube<float> *subcube;
    // arma::Cube<float>data=input->data();
    // std::shared_ptr<Tensor<float>> output ;
    // if (dimension == 1) {
    //     subdata = data.slice(0); // 提取第0维的指定索引的数据
    //     *subcube = arma::Cube<float>(subdata.memptr(), 1,input_h, input_w);
    //     output= std::make_shared<Tensor<float>>(subcube,1,input_h,input_w);

    // } else if (dimension == 2) {
    //     subdata=data.row(0); // 设置mat大小
    //     *subcube = arma::Cube<float>(subdata.memptr(),  input_c, 1,input_h); // 将 input_w 替换为 1
    //     output= std::make_shared<Tensor<float>>(subcube,input_h,1,input_c);
    // } else if (dimension == 3) {
    //     cout<<"Trouse 3"<<endl;
    //     subdata=data.col(0);
    //     *subcube = arma::Cube<float>(subdata.memptr(),  input_c,input_w,1 ); // 将 input_h 替换为 1
    //     output= std::make_shared<Tensor<float>>(subcube,input_w,input_c,1);
    // } else {
    //     cerr << "无效的维度！" << endl;
    //     return StatusCode::kInferInputsEmpty;
       
    // }
    //这里有问题。如果tensor像flatten操做一样都要使用cube为三维度数据的话，那么就不能用mat对象。
    
    outputs.at(i) = input;//不对数据进行处理，直接返回原数据
  }
  return StatusCode::kSuccess;
}

StatusCode UnbindLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& unbind_layer) {
  if (!op) {
    LOG(ERROR) << "The unbind operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the unbind layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (!op->has_parameter("dim"))  {
    LOG(ERROR) << "Can not find the dimension parameter";
    return StatusCode::kParseParameterError;
  }



  auto target_dim = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim"));
 

  if (target_dim == nullptr ) {
    LOG(ERROR) << "The start or end dimension parameter in the unbind layer is empty.";
    return StatusCode::kParseParameterError;
  }

  unbind_layer = std::make_shared<UnbindLayer>(target_dim->value);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kUnbindCreateInstance(UnbindLayer::CreateInstance, "torch.unbind");

}  // namespace kuiper_infer