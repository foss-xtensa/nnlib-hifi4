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
// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// gemmlowp.h: the main public interface header of gemmlowp.

#ifndef GEMMLOWP_PUBLIC_GEMMLOWP_H_
#define GEMMLOWP_PUBLIC_GEMMLOWP_H_
#include "../internal/dispatch_gemm_shape.h"
#include "bit_depth.h"
#include "map.h"
#include "output_stages.h"

namespace gemmlowp {

#ifndef HIFI_BUILD
class GemmContext : public MultiThreadGemmContext {};
#else
class GemmContext
{
    public:
    GemmContext()  {}
    ~GemmContext() {}
    //private:
        //int dummy;
};
#endif //HIFI_BUILD

// Computes a general matrix product ("GEMM").
// This is a version that supports per channel quantization.
template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename LhsOffset, typename RhsOffset, typename OutputPipelineType,
          typename GemmContextType>
void GemmWithOutputPipelinePC(GemmContextType* context,
                              const MatrixMap<const InputScalar, LhsOrder>& lhs,
                              const MatrixMap<const InputScalar, RhsOrder>& rhs,
                              MatrixMap<OutputScalar, ResultOrder>* result,
                              const LhsOffset& lhs_offset,
                              const RhsOffset& rhs_offset,
                              const OutputPipelineType& output_pipeline) {
  DispatchGemmShape<InputScalar, OutputScalar, BitDepthParams>(
      context, lhs, rhs, result, lhs_offset, rhs_offset, output_pipeline);
}

// Computes a general matrix product ("GEMM").
// This is the legacy version that does not support per channel quantization.
// The meaning of the offsets, result_mult_int and result_shift
// parameters is the same as in the standard EightBitIntGemm interface
// (which is also implemented in the eight_bit_int_gemm directory).
template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename OutputPipelineType, typename GemmContextType>
void GemmWithOutputPipeline(GemmContextType* context,
                            const MatrixMap<const InputScalar, LhsOrder>& lhs,
                            const MatrixMap<const InputScalar, RhsOrder>& rhs,
                            MatrixMap<OutputScalar, ResultOrder>* result,
                            int lhs_offset, int rhs_offset,
                            const OutputPipelineType& output_pipeline) {
  typedef VectorDup<const std::int32_t, VectorShape::Col> OffsetColDup;
  typedef VectorDup<const std::int32_t, VectorShape::Row> OffsetRowDup;
  const OffsetColDup lhs_offset_vector(lhs_offset, lhs.rows());
  const OffsetRowDup rhs_offset_vector(rhs_offset, rhs.cols());
  DispatchGemmShape<InputScalar, OutputScalar, BitDepthParams>(
      context, lhs, rhs, result, lhs_offset_vector, rhs_offset_vector,
      output_pipeline);
}

// Computes a general matrix product ("GEMM").
// The meaning of the offsets, result_mult_int and result_shift
// parameters is the same as in the standard EightBitIntGemm interface
// (which is also implemented in the eight_bit_int_gemm directory).
template <typename Scalar, typename BitDepthParams, MapOrder LhsOrder,
          MapOrder RhsOrder, MapOrder ResultOrder, typename GemmContextType>
void Gemm(GemmContextType* context,
          const MatrixMap<const Scalar, LhsOrder>& lhs,
          const MatrixMap<const Scalar, RhsOrder>& rhs,
          MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
          int rhs_offset, int result_offset, int result_mult_int,
          int result_shift) {
  GemmWithOutputPipeline<Scalar, Scalar, BitDepthParams>(
      context, lhs, rhs, result, lhs_offset, rhs_offset,
      MakeStandardOutputPipeline(result_offset, result_mult_int, result_shift));
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_GEMMLOWP_H_
