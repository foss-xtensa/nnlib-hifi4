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
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SVDF.h"

#ifndef HIFI_BUILD
#else
#include "xa_nnlib_ann_api.h"
#endif //HiFi_NNLIB_OPT

#include "CpuExecutor.h"

#ifndef HIFI_BUILD
#include "HalInterfaces.h"
#else
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#endif //HiFi_BUILD

namespace android {
namespace nn {

namespace {

template <typename T>
inline T *GetBuffer(RunTimeOperandInfo* operand) {
  return reinterpret_cast<T*>(operand->buffer);
}

template <typename T>
inline const T *GetBuffer(const RunTimeOperandInfo* operand) {
  return reinterpret_cast<const T*>(operand->buffer);
}

}

SVDF::SVDF(const Operation& operation,
           std::vector<RunTimeOperandInfo>& operands) {
    input_ = GetInput(operation, operands, kInputTensor);
    weights_feature_ = GetInput(operation, operands, kWeightsFeatureTensor);
    weights_time_ = GetInput(operation, operands, kWeightsTimeTensor);
    bias_ = GetInput(operation, operands, kBiasTensor);
    state_in_ = GetInput(operation, operands, kStateInTensor);

    params_.rank_ = getScalarData<int>(*GetInput(operation, operands, kRankParam));
#ifndef HIFI_BUILD
    params_.activation_ = static_cast<TfLiteFusedActivation>(getScalarData<int>(
#else
    params_.activation_ = static_cast<svdf_TfLiteFusedActivation>(getScalarData<int>(
#endif //HiFi_BUILD
        *GetInput(operation, operands, kActivationParam)));

    state_out_ = GetOutput(operation, operands, kStateOutTensor);
    output_ = GetOutput(operation, operands, kOutputTensor);
}

bool SVDF::Prepare(const Operation &operation,
                   std::vector<RunTimeOperandInfo> &operands,
                   Shape *stateShape,
                   Shape *outputShape) {
  // Check we have all the inputs and outputs we need.
  const int num_inputs = NumInputsWithValues(operation, operands);

  NN_CHECK(num_inputs == 6 || num_inputs == 7);
  NN_CHECK_EQ(NumOutputs(operation), 2);

  const RunTimeOperandInfo *input =
      GetInput(operation, operands, SVDF::kInputTensor);
  const RunTimeOperandInfo *weights_feature =
      GetInput(operation, operands, SVDF::kWeightsFeatureTensor);
  const RunTimeOperandInfo *weights_time =
      GetInput(operation, operands, SVDF::kWeightsTimeTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = getScalarData<int>(*GetInput(operation, operands, kRankParam));
  const uint32_t batch_size = SizeOfDimension(input, 0);
  const uint32_t num_filters = SizeOfDimension(weights_feature, 0);
  NN_CHECK_EQ(num_filters % rank, 0);
  const uint32_t num_units = num_filters / rank;
  const uint32_t memory_size = SizeOfDimension(weights_time, 1);
  NN_CHECK_EQ(SizeOfDimension(input, 1), SizeOfDimension(weights_feature, 1));
  NN_CHECK_EQ(SizeOfDimension(weights_time, 0), num_filters);

  const RunTimeOperandInfo *bias =
      GetInput(operation, operands, kBiasTensor);
  if (!IsNullInput(bias)) {
    NN_CHECK_EQ(SizeOfDimension(bias, 0), num_units);
  }

  // Resize state.
  const Shape &inputShape = input->shape();
  stateShape->type = inputShape.type;
#ifndef HIFI_BUILD
  stateShape->dimensions = { batch_size, memory_size * num_filters };
#else
  stateShape->dimensions.clear();
  stateShape->dimensions.push_back(batch_size);
  stateShape->dimensions.push_back(memory_size * num_filters);
#endif
  stateShape->offset = inputShape.offset;
  stateShape->scale = inputShape.scale;

  // Resize output.
  outputShape->type = inputShape.type;
#ifndef HIFI_BUILD
  outputShape->dimensions = { batch_size, num_units };
#else
  outputShape->dimensions.clear();
  outputShape->dimensions.push_back(batch_size);
  outputShape->dimensions.push_back(num_units);
#endif
  outputShape->offset = inputShape.offset;
  outputShape->scale = inputShape.scale;
#ifndef HIFI_WARNINGS
  (void)num_inputs;
#endif
  return true;
}

bool SVDF::Eval() {
    const int rank = params_.rank_;
    const int batch_size = SizeOfDimension(input_, 0);
    const int input_size = SizeOfDimension(input_, 1);
    const int num_filters = SizeOfDimension(weights_feature_, 0);
    const int num_units = num_filters / rank;
    const int memory_size = SizeOfDimension(weights_time_, 1);

    memcpy(GetBuffer<float>(state_out_), GetBuffer<float>(state_in_),
           sizeof(float) * batch_size * memory_size * num_filters);
    // Compute conv1d(inputs, weights_feature).
    for (int b = 0; b < batch_size; b++) {
        float* state_ptr_batch = GetBuffer<float>(state_out_) + b * memory_size * num_filters;
        for (int c = 0; c < num_filters; c++) {
            float* state_ptr = state_ptr_batch + c * memory_size;
            state_ptr[memory_size - 1] = 0.0;
        }
    }
    // The state left most column is used to save current cycle activation. This
    // is achieved by starting at state->data.f[memory_size - 1] and having the
    // stride equal to memory_size.
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        GetBuffer<float>(weights_feature_), num_filters, input_size,
        GetBuffer<float>(input_),  batch_size,
        &GetBuffer<float>(state_out_)[memory_size - 1], memory_size);

    // Compute matmul(state, weights_time).
    // The right most column is used to save temporary output (with the size of
    // num_filters). This is achieved by starting at state->data.f and having the
    // stride equal to memory_size.
    float scratch[batch_size * num_filters];
    for (int b = 0; b < batch_size; b++) {
        float* state_out_ptr_batch =
            GetBuffer<float>(state_out_) + b * memory_size * num_filters;
        float* scratch_ptr_batch = scratch + b * num_filters;
        tflite::tensor_utils::BatchVectorBatchVectorDotProduct(
            GetBuffer<float>(weights_time_), state_out_ptr_batch, memory_size, num_filters,
            scratch_ptr_batch, /*result_stride=*/1);
    }

    // Initialize output with bias if provided.
    if (!IsNullInput(bias_)) {
        tflite::tensor_utils::VectorBatchVectorAssign(
            GetBuffer<float>(bias_), num_units, batch_size,
            GetBuffer<float>(output_));
    } else {
        tflite::tensor_utils::ZeroVector(
            GetBuffer<float>(output_), batch_size * num_units);
    }

    // Reduction sum
    for (int b = 0; b < batch_size; b++) {
        float* output_ptr_batch = GetBuffer<float>(output_) + b * num_units;
        float* scratch_ptr_batch = scratch + b * num_filters;
        tflite::tensor_utils::ReductionSumVector(
            scratch_ptr_batch, output_ptr_batch, num_units, rank);
    }

    // Apply activation.
    for (int b = 0; b < batch_size; b++) {
        float* output_ptr_batch = GetBuffer<float>(output_) + b * num_units;
        tflite::tensor_utils::ApplyActivationToVector(
            output_ptr_batch, num_units,
#ifndef HIFI_BUILD
            params_.activation_, output_ptr_batch);
#else
            static_cast<TfLiteFusedActivation>(params_.activation_), output_ptr_batch);
#endif //HiFi_BUILD
    }

    // Right shift the state.
    for (int b = 0; b < batch_size; b++) {
        float* state_out_ptr_batch =
            GetBuffer<float>(state_out_) + b * memory_size * num_filters;
        for (int f = 0; f < num_filters; f++) {
            tflite::tensor_utils::VectorShiftLeft(state_out_ptr_batch, memory_size,
                                          /*shift_value=*/0.0);
            state_out_ptr_batch += memory_size;
        }
    }
    return true;
}

}  // namespace nn
}  // namespace android
