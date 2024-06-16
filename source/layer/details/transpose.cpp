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
#include "transpose.hpp"
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"



namespace kuiper_infer {
    using namespace arma;
TransposeLayer::TransposeLayer(int32_t start_dim, int32_t end_dim)
    : NonParamLayer("Transpose"), dim0(start_dim), dim1(end_dim) {}

StatusCode TransposeLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the transpose layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the transpose layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the transpose "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  int32_t start_dim = dim0-1;
  int32_t end_dim = dim1-1;
  int32_t total_dims = 4;  // NCHW






  if (start_dim < 0) {
    start_dim = total_dims + start_dim;
  }
  if (end_dim < 0) {
    end_dim = total_dims + end_dim;
  }

  dim0 = start_dim;
  dim1 = end_dim;
  if (end_dim <= start_dim) {
    LOG(ERROR) << "The end dim must greater than start dim";
    return StatusCode::kInferParameterError;
  }

  if (end_dim > 3 || start_dim < 0) {
    LOG(ERROR) << "The end dim must less than two and start dim must greater than zero";
    return StatusCode::kInferParameterError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);

    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input tensor array in the transpose layer has"
                    " an empty tensor "
                 << i << " th";
      return StatusCode::kInferInputsEmpty;
    }

    uword dim[3] = {input->channels(), input->rows(), input->cols()};
    uword newDim[3] = {dim[0], dim[1], dim[2]};
    // 交换维度
    newDim[start_dim] = dim[end_dim];
    newDim[end_dim] = dim[start_dim];

    arma::Cube<float>outputp (newDim[1], newDim[2],newDim[0]);
    arma::Cube<float> *ptr_output=&outputp;

    for (uword i = 0; i < dim[0]; ++i) {
        for (uword j = 0; j < dim[1]; ++j) {
            for (uword k = 0; k < dim[2]; ++k) {
                // 根据指定的 dim0 和 dim1 交换坐标
                uword coords[3] = {i, j, k};
                // cout<<i<<"_"<<j<<"_"<<k<<'_'<<endl;
                std::swap(coords[dim0], coords[dim1]);
                // cout<<coords[0]<<'_'<<coords[1]<<'_'<<coords[2]<<"*****"<<input->at(i, j, k)<<"****"<<i<<'_'<<j<<'_'<<k<<"\n";
                (*ptr_output)( coords[1], coords[2],coords[0]) = input->at(i, j, k);//T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) { this->data_.at(row, col, channel);
            }
        }
    }


    std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(ptr_output, newDim[0], newDim[1], newDim[2]);
    


    CHECK(input->size() == output->size()) << "The output and input shapes of the transpose layer do "
                                              "not match "
                                          << i << " th";

        



    
    outputs.at(i) = output;
    //delete ptr_output;

  }
  return StatusCode::kSuccess;
}

StatusCode TransposeLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& transpose_layer) {
  if (!op) {
    LOG(ERROR) << "The transpose operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the transpose layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (!(op->has_parameter("end_dim") or op->has_parameter("dim0")) ) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return StatusCode::kParseParameterError;
  }

  if (!(op->has_parameter("start_dim")or op->has_parameter("dim1"))) {//同时不具备start_dim和end_dim
    LOG(ERROR) << "Can not find the dimension parameter";
    return StatusCode::kParseParameterError;
  }

  auto start_dim = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim0"));
  auto end_dim = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim1"));

  if (start_dim == nullptr || end_dim == nullptr) {
    LOG(ERROR) << "The start or end dimension parameter in the transpose layer is empty.";
    return StatusCode::kParseParameterError;
  }

  transpose_layer = std::make_shared<TransposeLayer>(start_dim->value, end_dim->value);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kTransposeCreateInstance(TransposeLayer::CreateInstance, "torch.transpose");

}  // namespace kuiper_infer