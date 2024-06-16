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

// Created by fss on 22-12-21.
#include <gtest/gtest.h>
#include "data/tensor.hpp"
#include <iostream>
#include <armadillo>
#include <glog/logging.h>
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"
// #include "../../source/layer/details/transpose.hpp"
#include "../../source/layer/details/transpose.hpp"

using namespace kuiper_infer;
using namespace std;




using namespace arma;



// 打印3D张量的辅助函数
void printTensor11(const cube &tensor) {
    for (size_t i = 0; i < tensor.n_rows; ++i) {
        cout << "Slice " << i << ":" << endl;
        cout << tensor.slice(i) << endl;
    }
}

void printTensorDXY11(std::shared_ptr<Tensor<float>> &tensor) {
     const auto& raw_shapes = tensor->channels();
    // cout<<raw_shapes.at(0)<<" "<<raw_shapes.at(1)<<" "<<raw_shapes.at(2) <<"$$$$";
    for (size_t i = 0; i < raw_shapes; ++i) {
        cout << "Slice " << i << ":" << endl;
        cout << tensor->slice(i) << endl;
    }
}
void printTensorDXYout11(Tensor<float> tensor) {
     const auto& raw_shapes = tensor.channels();
    // cout<<raw_shapes.at(0)<<" "<<raw_shapes.at(1)<<" "<<raw_shapes.at(2) <<"$$$$";
    for (size_t i = 0; i < raw_shapes; ++i) {
        cout << "Slice " << i << ":" << endl;
        cout << tensor.slice(i) << endl;
    }
}



#if 1
TEST(test_Effnet, forward) {


    
  using namespace kuiper_infer;
  RuntimeGraph graph("/home/brown/GGB_kuiper/KuiperInfer/modelpath/Crib_effcient.pnnx.param",
                     "/home/brown/GGB_kuiper/KuiperInfer/modelpath/Crib_effcient.pnnx.bin");

  graph.Build();
  
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 500, 128);
    input->Fill(42.f);
    inputs.push_back(input);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(true);

  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
  // for (int i = 0; i < batch_size; ++i) {
  //   std::string file_path = "/home/brown/GGB_kuiper/KuiperInfer/modelpath/" + std::to_string((i + 1) * 10 + 1) + ".csv";
  //   const auto& output1 = CSVDataLoader::LoadData<float>(file_path);
  //   const auto& output2 = outputs.at(i);

  //   ASSERT_EQ(output1.size(), output2->size());
  //   for (int r = 0; r < output1.n_rows; ++r) {
  //     for (int c = 0; c < output1.n_cols; ++c) {
  //       ASSERT_LE(std::abs(output1.at(r, c) - output2->at(0, r, c)), 0.05)
  //           << " row: " << r << " col: " << c;
  //     }
  //   }
  // }

}
#endif




