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
#include "zeropad.hpp"
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"



namespace kuiper_infer {
    using namespace arma;
ZeropadLayer::ZeropadLayer(int32_t pad_top_, int32_t pad_bottom_, int32_t pad_left_, int32_t pad_right_)
    : NonParamLayer("Zeropad"), pad_top(pad_top_),pad_bottom(pad_bottom_),pad_left(pad_left_),pad_right(pad_right_) {}

StatusCode ZeropadLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
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

    int32_t pad_top1 = pad_top;
    int32_t pad_bottom1 = pad_bottom;
    int32_t pad_left1 = pad_left;
    int32_t pad_right1 = pad_right;

    const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) 
    {
        const std::shared_ptr<Tensor<float>>& input = inputs.at(i);

        if (input == nullptr || input->empty()) {
        LOG(ERROR) << "The input tensor array in the transpose layer has"
                        " an empty tensor "
                    << i << " th";
        return StatusCode::kInferInputsEmpty;
        }
    int original_rows = input->rows();
    int original_cols = input->cols();
    int original_channels = input->channels();
    int new_rows = original_rows + pad_top + pad_bottom;
    int new_cols = original_cols + pad_left + pad_right;

  // 创建填充后的立方体
    arma::Cube<float>output( new_rows, new_cols,original_channels, arma::fill::zeros);
    // cout<<output<<"@@@@@@@@@@@@@@@";
    arma::Cube<float> *out=&output;
    // output.reset();
    // 创建新的 Cube 对象
 // 复制原始数据到填充后的立方体中
    for (arma::uword c = 0; c < original_channels; ++c) 
    {
        // cout<<c<<"!!!!!!!!!\n";
        // 将subcube的值赋给临时变量sub
        // arma::Cube<float> sub = output.subcube( pad_top, pad_left, c, pad_top + original_rows - 1, pad_left + original_cols - 1,c);
    
        // 打印sub的值


        output.subcube(pad_top, pad_left, c ,pad_top + original_rows - 1, pad_left + original_cols - 1,c) = input->slice(c);
        // std::cout << "Subcube value:" << std::endl;
        // std::cout << sub << std::endl;
    }
    // 创建新的 Tensor 对象并返回
    std::shared_ptr<Tensor<float>> outputDXY = std::make_shared<Tensor<float>>(out, original_channels ,new_rows, new_cols);
    outputs.at(i) = outputDXY;

   
    }
    



  return StatusCode::kSuccess;
}

StatusCode ZeropadLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& zero_padding_layer) {
  if (!op) {
    LOG(ERROR) << "The transpose operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the transpose layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (!(  op->has_parameter("padding")) ) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return StatusCode::kParseParameterError;
  }

    cout<<"ZeroPadding Layer Parameters!!!"<<endl;
    // auto pt=params.at("padding");
  auto zero_padding_list_ptr = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));


  if (zero_padding_list_ptr == nullptr) {
    LOG(ERROR) << "The start or end dimension parameter in the ZeroPadding layer is empty.";
    return StatusCode::kParseParameterError;
  }

    if (zero_padding_list_ptr->value.size() != 4) {
        LOG(ERROR) << "The number of dimension parameter in the ZeroPadding layer is not right.";
        return StatusCode::kParseParameterError; 
    }


// std::for_each(zero_padding_list_ptr->value.begin(), zero_padding_list_ptr->value.end(), [](int num){
//     std::cout << num << " ";
// });
// std::cout << "ZeroPadding Layer Parameters!!!" << std::endl;
  zero_padding_layer = std::make_shared<ZeropadLayer>(zero_padding_list_ptr->value[0],zero_padding_list_ptr->value[1],zero_padding_list_ptr->value[2],zero_padding_list_ptr->value[3]);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kZeropadCreateInstance(ZeropadLayer::CreateInstance, "nn.ZeroPad2d");

}  // namespace kuiper_infer