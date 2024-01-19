/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
// Copyright 2017 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// dispatch_gemm_shape.h: dispatch GEMM calls according to their shape

#ifndef GEMMLOWP_INTERNAL_DISPATCH_GEMM_SHAPE_H_
#define GEMMLOWP_INTERNAL_DISPATCH_GEMM_SHAPE_H_

#ifndef HIFI_BUILD
#include "../internal/kernel_default.h"
#endif //HIFI_BUILD
#include "../public/map.h"
#include "../public/output_stages.h"
#ifndef HIFI_BUILD
#include "multi_thread_gemm.h"
#endif //HIFI_BUILD

namespace gemmlowp {

template <typename T>
struct TransposeImpl {
  typedef T DstType;
  static T Run(const T& t) { return t; }
};

template <typename T>
using TransposeType = typename TransposeImpl<T>::DstType;

template <typename T>
TransposeType<T> Transpose(const T& t) {
  return TransposeImpl<T>::Run(t);
}

template <MapOrder Order>
struct TransposeMapOrder {
  static constexpr MapOrder Value =
      Order == MapOrder::RowMajor ? MapOrder::ColMajor : MapOrder::RowMajor;
};

template <VectorShape Shape>
struct TransposeVectorShape {
  static constexpr VectorShape Value =
      Shape == VectorShape::Row ? VectorShape::Col : VectorShape::Row;
};

template <typename Scalar, VectorShape Shape>
struct TransposeImpl<VectorMap<Scalar, Shape>> {
  typedef VectorMap<Scalar, Shape> SrcType;
  static constexpr VectorShape TransposedShape =
      TransposeVectorShape<Shape>::Value;
  typedef VectorMap<Scalar, TransposedShape> DstType;
  static DstType Run(const SrcType& src) {
    return DstType(src.data(), src.size());
  }
};

template <typename Scalar, MapOrder Order>
struct TransposeImpl<MatrixMap<Scalar, Order>> {
  typedef MatrixMap<Scalar, Order> SrcType;
  static constexpr MapOrder TransposedOrder = TransposeMapOrder<Order>::Value;
  typedef MatrixMap<Scalar, TransposedOrder> DstType;
  static DstType Run(const SrcType& src) {
    return DstType(src.data(), src.cols(), src.rows(), src.stride());
  }
};

template <VectorShape Shape>
struct TransposeImpl<OutputStageQuantizeDownInt32ToUint8ScalePC<Shape>> {
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<Shape> SrcType;
  static const VectorShape TransposedShape = TransposeVectorShape<Shape>::Value;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<TransposedShape> DstType;
  static DstType Run(const SrcType& src) {
    DstType dst;
    dst.result_shift = src.result_shift;
    dst.result_offset = Transpose(src.result_offset);
    dst.result_mult_int = Transpose(src.result_mult_int);
    return dst;
  }
};

template <typename VectorMapType>
struct TransposeImpl<OutputStageBiasAddition<VectorMapType>> {
  typedef OutputStageBiasAddition<VectorMapType> SrcType;
  typedef TransposeType<VectorMapType> TransposedVectorMapType;
  typedef OutputStageBiasAddition<TransposedVectorMapType> DstType;
  static DstType Run(const SrcType& src) {
    DstType dst;
    dst.bias_vector = Transpose(src.bias_vector);
    return dst;
  }
};

// TODO(benoitjacob) - does anyone understand C++ variadic templates?
// How to use them to implement TransposeTuple? Note: there are lots
// of answers on StackOverflow but they seem to all involve either
// C++14/C++17 (we can only use C++11) or lots of abstract nonsense.
inline std::tuple<> TransposeTuple(const std::tuple<>& t) { return t; }

template <typename T0>
std::tuple<TransposeType<T0>> TransposeTuple(const std::tuple<T0>& t) {
  return std::make_tuple(Transpose(std::get<0>(t)));
}

template <typename T0, typename T1>
std::tuple<TransposeType<T0>, TransposeType<T1>> TransposeTuple(
    const std::tuple<T0, T1>& t) {
  return std::make_tuple(Transpose(std::get<0>(t)), Transpose(std::get<1>(t)));
}

template <typename T0, typename T1, typename T2>
std::tuple<TransposeType<T0>, TransposeType<T1>, TransposeType<T2>>
TransposeTuple(const std::tuple<T0, T1, T2>& t) {
  return std::make_tuple(Transpose(std::get<0>(t)), Transpose(std::get<1>(t)),
                         Transpose(std::get<2>(t)));
}

template <typename T0, typename T1, typename T2, typename T3>
std::tuple<TransposeType<T0>, TransposeType<T1>, TransposeType<T2>,
           TransposeType<T3>>
TransposeTuple(const std::tuple<T0, T1, T2, T3>& t) {
  return std::make_tuple(Transpose(std::get<0>(t)), Transpose(std::get<1>(t)),
                         Transpose(std::get<2>(t)), Transpose(std::get<3>(t)));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
std::tuple<TransposeType<T0>, TransposeType<T1>, TransposeType<T2>,
           TransposeType<T3>, TransposeType<T4>>
TransposeTuple(const std::tuple<T0, T1, T2, T3, T4>& t) {
  return std::make_tuple(Transpose(std::get<0>(t)), Transpose(std::get<1>(t)),
                         Transpose(std::get<2>(t)), Transpose(std::get<3>(t)),
                         Transpose(std::get<4>(t)));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5>
std::tuple<TransposeType<T0>, TransposeType<T1>, TransposeType<T2>,
           TransposeType<T3>, TransposeType<T4>, TransposeType<T5>>
TransposeTuple(const std::tuple<T0, T1, T2, T3, T4, T5>& t) {
  return std::make_tuple(Transpose(std::get<0>(t)), Transpose(std::get<1>(t)),
                         Transpose(std::get<2>(t)), Transpose(std::get<3>(t)),
                         Transpose(std::get<4>(t)), Transpose(std::get<5>(t)));
}

template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename LhsOffset, typename RhsOffset, typename OutputPipelineType,
          typename GemmContextType>
void DispatchGemmShape(GemmContextType* context,
                       const MatrixMap<const InputScalar, LhsOrder>& lhs,
                       const MatrixMap<const InputScalar, RhsOrder>& rhs,
                       MatrixMap<OutputScalar, ResultOrder>* result,
                       const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                       const OutputPipelineType& output_pipeline) {
  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  if (rows == 0 || cols == 0 || depth == 0) {
    // Vacuous GEMM, return early to avoid having to deal with
    // zero sizes below.
    return;
  }

  if (rows < cols) {
    auto transposed_result_map = Transpose(*result);
    return DispatchGemmShape<InputScalar, OutputScalar, BitDepthParams>(
        context, Transpose(rhs), Transpose(lhs), &transposed_result_map,
        Transpose(rhs_offset), Transpose(lhs_offset),
        TransposeTuple(output_pipeline));
  }

#ifndef HIFI_BUILD
  typedef DefaultKernel<BitDepthParams> Kernel;
  MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                  BitDepthParams>(context, Kernel(), lhs, rhs, result,
                                  lhs_offset, rhs_offset, output_pipeline);
#endif //HIFI_BUILD
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_DISPATCH_GEMM_SHAPE_H_
