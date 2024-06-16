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
cube transpose3D(const cube &input, uword dim0, uword dim1) {
    uword dim[3] = {input.n_rows, input.n_cols, input.n_slices};
    uword newDim[3] = {dim[0], dim[1], dim[2]};

    // 交换维度
    newDim[dim0] = dim[dim1];
    newDim[dim1] = dim[dim0];

    cube output(newDim[0], newDim[1], newDim[2]);

    for (uword i = 0; i < dim[0]; ++i) {
        for (uword j = 0; j < dim[1]; ++j) {
            for (uword k = 0; k < dim[2]; ++k) {
                // 根据指定的 dim0 和 dim1 交换坐标
                uword coords[3] = {i, j, k};
                std::swap(coords[dim0], coords[dim1]);
                cout<<coords[0]<<coords[1]<<coords[2]<<"----------------"<<i<<j<<k<<"\n";
                output(coords[0], coords[1], coords[2]) = input(i, j, k);
            }
        }
    }

    return output;
}






Tensor<float> transpose(std::vector<std::shared_ptr<Tensor<float>>>& inputs, uword dim0, uword dim1) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(0);
    uword dim[3] = {input->channels(), input->rows(), input->cols()};
    uword newDim[3] = {dim[0], dim[1], dim[2]};
    // 交换维度
    newDim[dim0] = dim[dim1];
    newDim[dim1] = dim[dim0];

    // 创建新的 Cube 对象
    arma::Cube<float>* ptr_output = new arma::Cube<float>(newDim[0], newDim[1], newDim[2]);
    for (uword i = 0; i < dim[0]; ++i) {
        for (uword j = 0; j < dim[1]; ++j) {
            for (uword k = 0; k < dim[2]; ++k) {
                // 根据指定的 dim0 和 dim1 交换坐标
                uword coords[3] = {i, j, k};
                std::swap(coords[dim0], coords[dim1]);
                cout<<coords[0]<<coords[1]<<coords[2]<<"*****"<<input->at(i, j, k)<<"****"<<i<<j<<k<<"\n";
                (*ptr_output)(coords[0], coords[1], coords[2]) = input->at(i, j, k);
            }
        }
    }
    // 创建新的 Tensor 对象并返回
    Tensor<float> outputDXY(ptr_output, newDim[0], newDim[1], newDim[2]);
    delete ptr_output;
    return outputDXY;
}

std::shared_ptr<Tensor<float>>  transpose_sharedptr(std::vector<std::shared_ptr<Tensor<float>>>& inputs, uword dim0, uword dim1) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(0);
    uword dim[3] = {input->channels(), input->rows(), input->cols()};
    uword newDim[3] = {dim[0], dim[1], dim[2]};
    // 交换维度
    newDim[dim0] = dim[dim1];
    newDim[dim1] = dim[dim0];

    // 创建新的 Cube 对象
    arma::Cube<float>* ptr_output = new arma::Cube<float>(newDim[0], newDim[1], newDim[2]);
    for (uword i = 0; i < dim[0]; ++i) {
        for (uword j = 0; j < dim[1]; ++j) {
            for (uword k = 0; k < dim[2]; ++k) {
                // 根据指定的 dim0 和 dim1 交换坐标
                uword coords[3] = {i, j, k};
                std::swap(coords[dim0], coords[dim1]);
                cout<<coords[0]<<coords[1]<<coords[2]<<"*****"<<input->at(i, j, k)<<"****"<<i<<j<<k<<"\n";
                (*ptr_output)(coords[0], coords[1], coords[2]) = input->at(i, j, k);
            }
        }
    }
    // 创建新的 Tensor 对象并返回
    std::shared_ptr<Tensor<float>> outputDXY = std::make_shared<Tensor<float>>(ptr_output, newDim[0], newDim[1], newDim[2]);
    delete ptr_output;
    return outputDXY;
}



// 打印3D张量的辅助函数
void printTensor(const cube &tensor) {
    for (size_t i = 0; i < tensor.n_rows; ++i) {
        cout << "Slice " << i << ":" << endl;
        cout << tensor.slice(i) << endl;
    }
}

void printTensorDXY(std::shared_ptr<Tensor<float>> &tensor) {
     const auto& raw_shapes = tensor->channels();
    // cout<<raw_shapes.at(0)<<" "<<raw_shapes.at(1)<<" "<<raw_shapes.at(2) <<"$$$$";
    for (size_t i = 0; i < raw_shapes; ++i) {
        cout << "Slice " << i << ":" << endl;
        cout << tensor->slice(i) << endl;
    }
}
void printTensorDXYout(Tensor<float> tensor) {
     const auto& raw_shapes = tensor.channels();
    // cout<<raw_shapes.at(0)<<" "<<raw_shapes.at(1)<<" "<<raw_shapes.at(2) <<"$$$$";
    for (size_t i = 0; i < raw_shapes; ++i) {
        cout << "Slice " << i << ":" << endl;
        cout << tensor.slice(i) << endl;
    }
}

// 


TEST(test_transpose,transpose2)
{
  #include <omp.h>

  const uint32_t channels = 2;
  const uint32_t rows = 3;
  const uint32_t cols = 3;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  
  input->RandN();
  cout << "Original Tensor:" << endl;
  printTensorDXY(input);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);


  Tensor<float> output=transpose(inputs,0,1);

  std::shared_ptr<Tensor<float>> DD=transpose_sharedptr(inputs,0,1) ;

     // 输出 Tensor 大小
    // std::cout << "Tensor size: " << output.size() << std::endl;
  cout << "Transposed Tensor:" << endl;
    printTensorDXYout(output);

    printTensorDXY(DD);


     int n = 10; // 循环的迭代次数
    int a[n];

    // 初始化数组
    for (int i = 0; i < n; ++i) {
        a[i] = i;
    }

    // 使用OpenMP并行化for循环
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int thread_id = omp_get_thread_num();
        a[i] = a[i] * 2;
        std::cout << "Iteration " << i << " is being executed by thread " << thread_id << std::endl;
    }

    // 输出结果
    for (int i = 0; i < n; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;


    cout << "END!!!!!!!!!!!!!!!!!!!!!!!!\n:" << endl;
    // // 创建一个3x3x3的张量
    // cube A(2, 3, 3, fill::randu);

    // // 打印原始张量


    // // 交换第 0 维和第 1 维
    // cube B = transpose3D(A, 1, 2);

    // 打印转置后的张量



}


#if 0
TEST(test_transpose, forward) {
  using namespace kuiper_infer;
  RuntimeGraph graph("/home/brown/GGB_kuiper/KuiperInfer/modelpath/Crib_effcient.pnnx.param",
                     "/home/brown/GGB_kuiper/KuiperInfer/modelpath/Crib_effcient.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 500, 128);
    input->Fill(42.f);
    inputs.push_back(input);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
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




TEST(test_transpose, forward_transpose2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 500, 128);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  TransposeLayer transpose_layer(0,1);
  const auto status = transpose_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  // for (int i = 0; i < inputs.size(); ++i) {
  //   std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
  //   input_->Transform([](const float f) {
  //     if (f < 0) {
  //       return 0.f;
  //     } else {
  //       return f;
  //     }
  //   });
  //   std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
  //   CHECK(input_->size() == output_->size());
  //   uint32_t size = input_->size();
  //   for (uint32_t j = 0; j < size; ++j) {
  //     ASSERT_EQ(output_->index(j), input_->index(j));
  //   }
  // }
}


std::shared_ptr<Tensor<float>>  zeroPad2d_ptr(std::vector<std::shared_ptr<Tensor<float>>>& inputs, int pad_top, int pad_bottom, int pad_left, int pad_right) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(0);
    // uword dim[3] = {input->channels(), input->rows(), input->cols()};
    
    int original_rows = input->rows();
    int original_cols = input->cols();
    int original_channels = input->channels();
    int new_rows = original_rows + pad_top + pad_bottom;
    int new_cols = original_cols + pad_left + pad_right;

    // uword newDim[3] = {dim[0], dim[1], dim[2]};
    cout<<original_channels<<"__"<<new_rows<<"__"<<new_cols<<"\n" ;
    // 创建填充后的立方体
    arma::Cube<float>output( new_rows, new_cols,original_channels, arma::fill::zeros);
    // cout<<output<<"@@@@@@@@@@@@@@@";
    arma::Cube<float> *out=&output;
    // output.reset();
    // 创建新的 Cube 对象
 // 复制原始数据到填充后的立方体中
    for (arma::uword c = 0; c < original_channels; ++c) {
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
    // delete out;
    return outputDXY;
}

arma::Mat<float> zeroPad2d(const arma::Mat<float>& input, int pad_top, int pad_bottom, int pad_left, int pad_right) {
    int original_rows = input.n_rows;
    int original_cols = input.n_cols;
    int new_rows = original_rows + pad_top + pad_bottom;
    int new_cols = original_cols + pad_left + pad_right;

    // Create a new matrix with the new size and fill it with zeros
    arma::Mat<float> padded(new_rows, new_cols, arma::fill::zeros);

    // Copy the original matrix into the new matrix with the specified padding
    padded.submat(pad_top, pad_left, pad_top + original_rows - 1, pad_left + original_cols - 1) = input;

    return padded;
}

TEST(test_transpose, padding) {

// // Function to pad a matrix with zeros

// // Create a 3x3 input matrix
//     arma::Mat<float> input = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 9}
//     };

//     std::cout << "Original Matrix:" << std::endl;
//     input.print();

//    
//     arma::Mat<float> padded = zeroPad2d(input, 1, 1, 0, 0);

//     std::cout << "Padded Matrix:" << std::endl;
//     padded.print();

    const uint32_t channels = 2;
    const uint32_t rows = 3;
    const uint32_t cols = 3;

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);



    input->RandN();
    cout << "Original Tensor:" << endl;
    printTensorDXY(input);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
    std::shared_ptr<Tensor<float>> out=zeroPad2d_ptr(inputs,1,0,1,0); // Apply zero padding (pad_top = 1, pad_bottom = 1, pad_left = 0, pad_right = 0)
    printTensorDXY(out);
}

std::shared_ptr<Tensor<float>>  unbind_ptr(std::vector<std::shared_ptr<Tensor<float>>>& inputs, int target_dim) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(0);
    // uword dim[3] = {input->channels(), input->rows(), input->cols()};
    
    int original_rows = input->rows();
    int original_cols = input->cols();
    int original_channels = input->channels();

    cout<<original_channels<<original_rows<<original_cols<<"!!!!!!"<<endl;


    // 创建填充后的立方体
    arma::Cube<float>output( original_rows,original_cols,original_channels, arma::fill::zeros);
    // cout<<output<<"@@@@@@@@@@@@@@@";
    arma::Cube<float> *out=&output;
    // output.reset();
    // 创建新的 Cube 对象
 // 复制原始数据到填充后的立方体中
        // 取出第3和第4维度的部分

    arma::Cube<float> data=input->data();
        // 指定维度和索引
    int dimension = 1; // 第三个维度
    int index = 1;     // 索引为1

    // 提取指定维度上的数据
    arma::Mat<float> subdata;

    if (dimension == 0) {
        subdata = data.slice(index); // 提取第0维的指定索引的数据
    } else if (dimension == 1) {
        subdata=data.row(0); // 设置mat大小

    } else if (dimension == 2) {
        subdata=data.col(0);
    } else {
        cerr << "无效的维度！" << endl;
       
    }
    // 输出取出的部分
    cout << "取出的部分:\n" << subdata << endl;
    // 创建新的 Tensor 对象并返回
    std::shared_ptr<Tensor<float>> outputDXY = std::make_shared<Tensor<float>>(out, original_channels ,original_rows, original_cols);
    // delete out;
    return outputDXY;
}

TEST(test_transpose, unbind) {

// // Function to pad a matrix with zeros

// // Create a 3x3 input matrix
//     arma::Mat<float> input = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 9}
//     };

//     std::cout << "Original Matrix:" << std::endl;
//     input.print();

//    
//     arma::Mat<float> padded = zeroPad2d(input, 1, 1, 0, 0);

//     std::cout << "Padded Matrix:" << std::endl;
//     padded.print();

    const uint32_t channels = 2;
    const uint32_t rows = 3;
    const uint32_t cols = 3;

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);



    input->RandN();
    cout << "Original Tensor:" << endl;
    printTensorDXY(input);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
    std::shared_ptr<Tensor<float>> out=unbind_ptr(inputs,3); // Apply zero padding (pad_top = 1, pad_bottom = 1, pad_left = 0, pad_right = 0)
    printTensorDXY(out);
}



std::shared_ptr<Tensor<float>>  sum_ptr(std::vector<std::shared_ptr<Tensor<float>>>& inputs, int target_dim) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(0);
    // uword dim[3] = {input->channels(), input->rows(), input->cols()};
    
    int original_rows = input->rows();
    int original_cols = input->cols();
    int original_channels = input->channels();

    cout<<original_channels<<original_rows<<original_cols<<"!!!!!!"<<endl;



    // output.reset();
    // 创建新的 Cube 对象
 // 复制原始数据到填充后的立方体中
        // 取出第3和第4维度的部分

    arma::Cube<float> data=input->data();
        // arma::Cube<float>data=input->data();
    arma::Cube<float> sum_result;
    arma::Cube<float> *out;
    // std::shared_ptr<Tensor<float>> output ;

            switch(target_dim) {
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
                    
            }
    cout<<sum_result<<"!!!!!!"<<sum_result.n_slices<<sum_result.n_rows<<sum_result.n_cols<<endl;
    out=&sum_result;
    std::shared_ptr<Tensor<float>> outputDXY = std::make_shared<Tensor<float>>(out, sum_result.n_slices  ,sum_result.n_rows,sum_result.n_cols);
    return outputDXY;
           
}


TEST(test_transpose, tsum) {

// // Function to pad a matrix with zeros

// // Create a 3x3 input matrix
//     arma::Mat<float> input = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 9}
//     };

//     std::cout << "Original Matrix:" << std::endl;
//     input.print();

//    
//     arma::Mat<float> padded = zeroPad2d(input, 1, 1, 0, 0);

//     std::cout << "Padded Matrix:" << std::endl;
//     padded.print();

    const uint32_t channels = 2;
    const uint32_t rows = 3;
    const uint32_t cols = 4;

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);



    input->RandN();
    cout << "Test Summing!!!!!!!!Original Tensor:" << endl;
    printTensorDXY(input);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
    std::shared_ptr<Tensor<float>> out=sum_ptr(inputs,0); // Apply zero padding (pad_top = 1, pad_bottom = 1, pad_left = 0, pad_right = 0)
    printTensorDXY(out);
}