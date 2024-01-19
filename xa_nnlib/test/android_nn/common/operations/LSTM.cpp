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

#include "LSTM.h"

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

}  // anonymous namespace

LSTMCell::LSTMCell(const Operation& operation,
                   std::vector<RunTimeOperandInfo>& operands) {
  input_ = GetInput(operation, operands, kInputTensor);

  input_to_input_weights_ = GetInput(operation, operands, kInputToInputWeightsTensor);  // optional
  input_to_forget_weights_ = GetInput(operation, operands, kInputToForgetWeightsTensor);
  input_to_cell_weights_ = GetInput(operation, operands, kInputToCellWeightsTensor);
  input_to_output_weights_ = GetInput(operation, operands, kInputToOutputWeightsTensor);

  recurrent_to_input_weights_ =
      GetInput(operation, operands, kRecurrentToInputWeightsTensor);  // optional
  recurrent_to_forget_weights_ = GetInput(operation, operands, kRecurrentToForgetWeightsTensor);
  recurrent_to_cell_weights_ = GetInput(operation, operands, kRecurrentToCellWeightsTensor);
  recurrent_to_output_weights_ = GetInput(operation, operands, kRecurrentToOutputWeightsTensor);

  cell_to_input_weights_ = GetInput(operation, operands, kCellToInputWeightsTensor);    // optional
  cell_to_forget_weights_ = GetInput(operation, operands, kCellToForgetWeightsTensor);  // optional
  cell_to_output_weights_ = GetInput(operation, operands, kCellToOutputWeightsTensor);  // optional

  input_gate_bias_ = GetInput(operation, operands, kInputGateBiasTensor);
  forget_gate_bias_ = GetInput(operation, operands, kForgetGateBiasTensor);
  cell_bias_ = GetInput(operation, operands, kCellGateBiasTensor);
  output_gate_bias_ = GetInput(operation, operands, kOutputGateBiasTensor);

  projection_weights_ = GetInput(operation, operands, kProjectionWeightsTensor);  // optional
  projection_bias_ = GetInput(operation, operands, kProjectionBiasTensor);        // optional

  output_state_in_ = GetInput(operation, operands, kOutputStateInTensor);
  cell_state_in_ = GetInput(operation, operands, kCellStateInTensor);

#ifndef HIFI_BUILD
  params_.activation_ = static_cast<TfLiteFusedActivation>(getScalarData<int32_t>(
#else
  params_.activation_ = static_cast<lstm_TfLiteFusedActivation>(getScalarData<int32_t>(
#endif //HiFi_BUILD
      *GetInput(operation, operands, kActivationParam)));
  params_.cell_clip_ = getScalarData<float>(*GetInput(operation, operands, kCellClipParam));
  params_.proj_clip_ = getScalarData<float>(*GetInput(operation, operands, kProjClipParam));

  output_state_out_ = GetOutput(operation, operands, kOutputStateOutTensor);
  cell_state_out_ = GetOutput(operation, operands, kCellStateOutTensor);
  output_ = GetOutput(operation, operands, kOutputTensor);

  scratch_buffer_ = GetOutput(operation, operands, kScratchBufferTensor);
}

bool LSTMCell::CheckInputTensorDimensions(
    const Operation &operation, std::vector<RunTimeOperandInfo> &operands,
    uint32_t n_input, uint32_t n_output, uint32_t n_cell) {
  LSTMParams params = {
#ifndef HIFI_BUILD
    .activation_ = static_cast<TfLiteFusedActivation>(getScalarData<int32_t>(
#else
    .activation_ = static_cast<lstm_TfLiteFusedActivation>(getScalarData<int32_t>(
#endif //HiFi_BUILD
        *GetInput(operation, operands, LSTMCell::kActivationParam))),
    .cell_clip_  = getScalarData<float>(*GetInput(operation, operands, LSTMCell::kCellClipParam)),
    .proj_clip_  = getScalarData<float>(*GetInput(operation, operands, LSTMCell::kProjClipParam))
  };

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  NN_CHECK(params.cell_clip_ >= 0);
  NN_CHECK(params.proj_clip_ >= 0);

  const RunTimeOperandInfo *input_to_input_weights =
      GetInput(operation, operands, LSTMCell::kInputToInputWeightsTensor);
  if (!IsNullInput(input_to_input_weights)) {
    NN_CHECK_EQ(NumDimensions(input_to_input_weights), 2);
    NN_CHECK_EQ(SizeOfDimension(input_to_input_weights, 0), n_cell);
    NN_CHECK_EQ(SizeOfDimension(input_to_input_weights, 1), n_input);
  }

  const RunTimeOperandInfo *input_to_forget_weights =
      GetInput(operation, operands, LSTMCell::kInputToForgetWeightsTensor);
  NN_CHECK_EQ(NumDimensions(input_to_forget_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(input_to_forget_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(input_to_forget_weights, 1), n_input);

  const RunTimeOperandInfo *input_to_cell_weights =
      GetInput(operation, operands, LSTMCell::kInputToCellWeightsTensor);
  NN_CHECK_EQ(NumDimensions(input_to_cell_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(input_to_cell_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(input_to_cell_weights, 1), n_input);

  const RunTimeOperandInfo *recurrent_to_input_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToInputWeightsTensor);
  if (!IsNullInput(recurrent_to_input_weights)) {
    NN_CHECK_EQ(NumDimensions(recurrent_to_input_weights), 2);
    NN_CHECK_EQ(SizeOfDimension(recurrent_to_input_weights, 0), n_cell);
    NN_CHECK_EQ(SizeOfDimension(recurrent_to_input_weights, 1), n_output);
  }

  const RunTimeOperandInfo *recurrent_to_forget_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToForgetWeightsTensor);
  NN_CHECK_EQ(NumDimensions(recurrent_to_forget_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_forget_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_forget_weights, 1), n_output);

  const RunTimeOperandInfo *recurrent_to_cell_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToCellWeightsTensor);
  NN_CHECK_EQ(NumDimensions(recurrent_to_cell_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_cell_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_cell_weights, 1), n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
      (!IsNullInput(input_to_input_weights) &&
       !IsNullInput(recurrent_to_input_weights)) ||
      (IsNullInput(input_to_input_weights) &&
       IsNullInput(recurrent_to_input_weights));
  NN_CHECK(cifg_weights_all_or_none);

  const RunTimeOperandInfo *cell_to_input_weights =
      GetInput(operation, operands, LSTMCell::kCellToInputWeightsTensor);
  if (!IsNullInput(cell_to_input_weights)) {
    NN_CHECK_EQ(NumDimensions(cell_to_input_weights), 1);
    NN_CHECK_EQ(SizeOfDimension(cell_to_input_weights, 0), n_cell);
  }

  const RunTimeOperandInfo *cell_to_forget_weights =
      GetInput(operation, operands, LSTMCell::kCellToForgetWeightsTensor);
  if (!IsNullInput(cell_to_forget_weights)) {
    NN_CHECK_EQ(NumDimensions(cell_to_forget_weights), 1);
    NN_CHECK_EQ(SizeOfDimension(cell_to_forget_weights, 0), n_cell);
  }

  const RunTimeOperandInfo *cell_to_output_weights =
      GetInput(operation, operands, LSTMCell::kCellToOutputWeightsTensor);
  if (!IsNullInput(cell_to_output_weights)) {
    NN_CHECK_EQ(NumDimensions(cell_to_output_weights), 1);
    NN_CHECK_EQ(SizeOfDimension(cell_to_output_weights, 0), n_cell);
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = IsNullInput(input_to_input_weights);
  const bool peephole_weights_all_or_none =
      ((!IsNullInput(cell_to_input_weights) || use_cifg) &&
       !IsNullInput(cell_to_forget_weights) &&
       !IsNullInput(cell_to_output_weights)) ||
      (IsNullInput(cell_to_input_weights) &&
       IsNullInput(cell_to_forget_weights) &&
       IsNullInput(cell_to_output_weights));
  NN_CHECK(peephole_weights_all_or_none);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  const RunTimeOperandInfo* input_gate_bias =
      GetInput(operation, operands, LSTMCell::kInputGateBiasTensor);
  if (use_cifg) {
    NN_CHECK(IsNullInput(input_gate_bias));
  } else {
    NN_CHECK_EQ(NumDimensions(input_gate_bias), 1);
    NN_CHECK_EQ(SizeOfDimension(input_gate_bias, 0), n_cell);
  }

  const RunTimeOperandInfo *forget_gate_bias =
      GetInput(operation, operands, LSTMCell::kForgetGateBiasTensor);
  NN_CHECK_EQ(NumDimensions(forget_gate_bias), 1);
  NN_CHECK_EQ(SizeOfDimension(forget_gate_bias, 0), n_cell);

  const RunTimeOperandInfo *cell_bias =
      GetInput(operation, operands, LSTMCell::kCellGateBiasTensor);
  NN_CHECK_EQ(NumDimensions(cell_bias), 1);
  NN_CHECK_EQ(SizeOfDimension(cell_bias, 0), n_cell);

  const RunTimeOperandInfo *output_gate_bias =
      GetInput(operation, operands, LSTMCell::kOutputGateBiasTensor);
  NN_CHECK_EQ(NumDimensions(output_gate_bias), 1);
  NN_CHECK_EQ(SizeOfDimension(output_gate_bias, 0), n_cell);

  const RunTimeOperandInfo *projection_weights =
      GetInput(operation, operands, LSTMCell::kProjectionWeightsTensor);
  if (!IsNullInput(projection_weights)) {
    NN_CHECK_EQ(NumDimensions(projection_weights), 2);
    NN_CHECK_EQ(SizeOfDimension(projection_weights, 0), n_output);
    NN_CHECK_EQ(SizeOfDimension(projection_weights, 1), n_cell);
  }

  const RunTimeOperandInfo *projection_bias =
      GetInput(operation, operands, LSTMCell::kProjectionBiasTensor);
  if (!IsNullInput(projection_bias)) {
    NN_CHECK_EQ(NumDimensions(projection_bias), 1);
    NN_CHECK_EQ(SizeOfDimension(projection_bias, 0), n_output);
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  // TODO: make sure this is correct.
  const bool projecton_tensors_consistent =
      (!IsNullInput(projection_weights) || IsNullInput(projection_bias));
  NN_CHECK(projecton_tensors_consistent == true);
#ifndef HIFI_WARNINGS
    (void)peephole_weights_all_or_none;
    (void)output_gate_bias;
    (void)input_to_forget_weights;
    (void)input_gate_bias;
    (void)recurrent_to_forget_weights;
    (void)cifg_weights_all_or_none;
    (void)projecton_tensors_consistent;
    (void)forget_gate_bias;
    (void)params;
    (void)input_to_cell_weights;
    (void)cell_bias;
    (void)recurrent_to_cell_weights;
#endif
  return true;
}

bool LSTMCell::Prepare(const Operation &operation,
                       std::vector<RunTimeOperandInfo> &operands,
                       Shape *scratchShape,
                       Shape *outputStateShape,
                       Shape *cellStateShape,
                       Shape *outputShape) {
  // Check we have all the inputs and outputs we need.
  NN_CHECK(NumInputsWithValues(operation, operands) >= 15 &&
           NumInputsWithValues(operation, operands) <= 23);
  NN_CHECK_EQ(NumOutputs(operation), 4);

  // Inferring batch size, number of outputs and number of cells from the
  // input tensors.
  const RunTimeOperandInfo *input =
      GetInput(operation, operands, LSTMCell::kInputTensor);
  NN_CHECK(NumDimensions(input) > 1);
  const uint32_t n_batch = SizeOfDimension(input, 0);
  const uint32_t n_input = SizeOfDimension(input, 1);

  const RunTimeOperandInfo *input_to_output_weights =
      GetInput(operation, operands, LSTMCell::kInputToOutputWeightsTensor);
  const uint32_t n_cell = SizeOfDimension(input_to_output_weights, 0);
  NN_CHECK_EQ(NumDimensions(input_to_output_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(input_to_output_weights, 1), n_input);

  const RunTimeOperandInfo *recurrent_to_output_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToOutputWeightsTensor);
  NN_CHECK_EQ(NumDimensions(recurrent_to_output_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_output_weights, 0),
                    n_cell);
  const uint32_t n_output = SizeOfDimension(recurrent_to_output_weights, 1);

  // Check that input tensor dimensions matches with each other.
  if (!CheckInputTensorDimensions(operation, operands, n_input, n_output, n_cell)) {
    return false;
  }

  // Resize the output and output_state tensors.
  const Shape &inputShape = input->shape();

  outputShape->type = inputShape.type;
#ifndef HIFI_BUILD
  outputShape->dimensions = { n_batch, n_output };
#else
  outputShape->dimensions.clear();
  outputShape->dimensions.push_back(n_batch);
  outputShape->dimensions.push_back(n_output);
#endif
  outputShape->offset = inputShape.offset;
  outputShape->scale = inputShape.scale;

  outputStateShape->type = inputShape.type;
#ifndef HIFI_BUILD
  outputStateShape->dimensions = { n_batch, n_output };
#else
  outputStateShape->dimensions.clear();
  outputStateShape->dimensions.push_back(n_batch);
  outputStateShape->dimensions.push_back(n_output);
#endif
  outputStateShape->offset = inputShape.offset;
  outputStateShape->scale = inputShape.scale;

  cellStateShape->type = inputShape.type;
#ifndef HIFI_BUILD
  cellStateShape->dimensions = { n_batch, n_cell };
#else
  cellStateShape->dimensions.clear();
  cellStateShape->dimensions.push_back(n_batch);
  cellStateShape->dimensions.push_back(n_cell);
#endif
  cellStateShape->offset = inputShape.offset;
  cellStateShape->scale = inputShape.scale;

  const RunTimeOperandInfo *input_to_input_weights =
      GetInput(operation, operands, LSTMCell::kInputToInputWeightsTensor);
  const bool use_cifg = IsNullInput(input_to_input_weights);
#ifndef HIFI_BUILD
  if (use_cifg) {
    // Reserving space for Cell, Forget, Output gates
    scratchShape->dimensions = { n_batch, n_cell * 3 };
  } else {
    // Reserving space for Input, Cell, Forget, Output gates
    scratchShape->dimensions = { n_batch, n_cell * 4 };
  }
#else
  if (use_cifg) {
    // Reserving space for Cell, Forget, Output gates
    scratchShape->dimensions.clear();
    scratchShape->dimensions.push_back(n_batch); 
    scratchShape->dimensions.push_back(n_cell * 3);
  } else {
    // Reserving space for Input, Cell, Forget, Output gates
    scratchShape->dimensions.clear();
    scratchShape->dimensions.push_back(n_batch); 
    scratchShape->dimensions.push_back(n_cell * 4);
  }
#endif
  scratchShape->type = inputShape.type;
  scratchShape->offset = inputShape.offset;
  scratchShape->scale = inputShape.scale;

  return true;
}

bool LSTMCell::Eval() {
  const int32_t n_batch = input_->shape().dimensions[0];
  const uint32_t n_input = input_->shape().dimensions[1];
  // n_cell and n_output will be the same size when there is no projection.
  const uint32_t n_cell = input_to_output_weights_->shape().dimensions[0];
  const uint32_t n_output = recurrent_to_output_weights_->shape().dimensions[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_->lifetime == OperandLifeTime::NO_VALUE);
  const bool use_peephole = (cell_to_output_weights_->lifetime != OperandLifeTime::NO_VALUE);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = reinterpret_cast<float*>(scratch_buffer_->buffer);
    forget_gate_scratch = cell_scratch + n_cell * n_batch;
    output_gate_scratch = cell_scratch + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = reinterpret_cast<float*>(scratch_buffer_->buffer);
    cell_scratch = input_gate_scratch + n_cell * n_batch;
    forget_gate_scratch = input_gate_scratch + 2 * n_cell * n_batch;
    output_gate_scratch = input_gate_scratch + 3 * n_cell * n_batch;
  }

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
  // Initialize scratch buffers with bias.
  if (!use_cifg) {
    tflite::tensor_utils::VectorBatchVectorAssign(GetBuffer<float>(input_gate_bias_),
                                                  n_cell, n_batch, input_gate_scratch);
  }
  tflite::tensor_utils::VectorBatchVectorAssign(GetBuffer<float>(forget_gate_bias_),
                                                n_cell, n_batch, forget_gate_scratch);
  tflite::tensor_utils::VectorBatchVectorAssign(GetBuffer<float>(cell_bias_),
                                                n_cell, n_batch, cell_scratch);
  tflite::tensor_utils::VectorBatchVectorAssign(GetBuffer<float>(output_gate_bias_),
                                                n_cell, n_batch, output_gate_scratch);

  // For each batch and cell: compute input_weight * input.
  if (!use_cifg) {
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        GetBuffer<float>(input_to_input_weights_), n_cell, n_input,
        GetBuffer<float>(input_), n_batch, input_gate_scratch, /*result_stride*/1);
  }
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(input_to_forget_weights_), n_cell, n_input,
      GetBuffer<float>(input_), n_batch, forget_gate_scratch, /*result_stride*/1);
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(input_to_cell_weights_), n_cell, n_input,
      GetBuffer<float>(input_), n_batch, cell_scratch, /*result_stride*/1);
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(input_to_output_weights_), n_cell, n_input,
      GetBuffer<float>(input_), n_batch, output_gate_scratch, /*result_stride*/1);

  // For each batch and cell: compute recurrent_weight * output_state.
  if (!use_cifg) {
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        GetBuffer<float>(recurrent_to_input_weights_), n_cell, n_output,
        GetBuffer<float>(output_state_in_), n_batch, input_gate_scratch, /*result_stride*/1);
  }
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(recurrent_to_forget_weights_), n_cell, n_output,
      GetBuffer<float>(output_state_in_), n_batch, forget_gate_scratch, /*result_stride*/1);
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(recurrent_to_cell_weights_), n_cell, n_output,
      GetBuffer<float>(output_state_in_), n_batch, cell_scratch, /*result_stride*/1);
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(recurrent_to_output_weights_), n_cell, n_output,
      GetBuffer<float>(output_state_in_), n_batch, output_gate_scratch, /*result_stride*/1);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole) {
      tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          GetBuffer<float>(cell_to_input_weights_), n_cell,
          GetBuffer<float>(cell_state_in_), n_batch, input_gate_scratch);
    }
    tflite::tensor_utils::ApplySigmoidToVector(input_gate_scratch,
                                               n_cell * n_batch,
                                               input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole) {
    tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        GetBuffer<float>(cell_to_forget_weights_), n_cell,
        GetBuffer<float>(cell_state_in_), n_batch, forget_gate_scratch);
  }
  tflite::tensor_utils::ApplySigmoidToVector(forget_gate_scratch,
                                             n_cell * n_batch,
                                             forget_gate_scratch);

  // For each batch and cell: update the cell.
  tflite::tensor_utils::VectorVectorCwiseProduct(
      forget_gate_scratch, GetBuffer<float>(cell_state_in_), n_batch * n_cell,
      GetBuffer<float>(cell_state_out_));
  tflite::tensor_utils::ApplyActivationToVector(
      cell_scratch, n_batch * n_cell,
#ifndef HIFI_BUILD
      params_.activation_, cell_scratch);
#else
      static_cast<TfLiteFusedActivation>(params_.activation_), cell_scratch);
#endif //HiFi_BUILD
  if (use_cifg) {
    tflite::tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                                     forget_gate_scratch);
    tflite::tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, forget_gate_scratch, n_batch * n_cell,
        GetBuffer<float>(cell_state_out_));
  } else {
    tflite::tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, input_gate_scratch, n_batch * n_cell,
        GetBuffer<float>(cell_state_out_));
  }
  if (params_.cell_clip_ > 0.0) {
    tflite::tensor_utils::ClipVector(GetBuffer<float>(cell_state_out_), n_batch * n_cell,
                                     params_.cell_clip_, GetBuffer<float>(cell_state_out_));
  }

  // For each batch and cell: update the output gate.
  if (use_peephole) {
    tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        GetBuffer<float>(cell_to_output_weights_), n_cell,
        GetBuffer<float>(cell_state_out_), n_batch, output_gate_scratch);
  }
  tflite::tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                             output_gate_scratch);
  tflite::tensor_utils::ApplyActivationToVector(GetBuffer<float>(cell_state_out_),
                                                n_batch * n_cell,
#ifndef HIFI_BUILD
                                                params_.activation_,
#else
                                                static_cast<TfLiteFusedActivation>(params_.activation_),
#endif //HiFi_BUILD
                                                cell_scratch);
  tflite::tensor_utils::VectorVectorCwiseProduct(output_gate_scratch,
                                                 cell_scratch, n_batch * n_cell,
                                                 output_gate_scratch);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight =
          (projection_weights_->lifetime != OperandLifeTime::NO_VALUE);
  const bool use_projection_bias = (projection_bias_->lifetime != OperandLifeTime::NO_VALUE);
  if (use_projection_weight) {
    if (use_projection_bias) {
      tflite::tensor_utils::VectorBatchVectorAssign(GetBuffer<float>(projection_bias_), n_output,
                                                    n_batch, GetBuffer<float>(output_));
    } else {
      tflite::tensor_utils::ZeroVector(GetBuffer<float>(output_), n_batch * n_output);
    }
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        GetBuffer<float>(projection_weights_), n_output, n_cell,
        output_gate_scratch, n_batch, GetBuffer<float>(output_),
        /*result_stride*/1);
    if (params_.proj_clip_ > 0.0) {
      tflite::tensor_utils::ClipVector(GetBuffer<float>(output_), n_batch * n_output,
                               params_.proj_clip_, GetBuffer<float>(output_));
    }
  } else {
    tflite::tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                             GetBuffer<float>(output_));
  }
  tflite::tensor_utils::CopyVector(GetBuffer<float>(output_), n_batch * n_output,
                           GetBuffer<float>(output_state_out_));
#else
  int b, ret = 0;
  if (!use_cifg) {
    for(b = 0; b < n_batch; b++)
    {
      ret = xa_nn_matXvec_f32xf32_f32(
          input_gate_scratch+b*n_cell,
          GetBuffer<float>(input_to_input_weights_), GetBuffer<float>(recurrent_to_input_weights_),
          GetBuffer<float>(input_)+b*n_input, GetBuffer<float>(output_state_in_)+b*n_output,
          GetBuffer<float>(input_gate_bias_), n_cell, n_input, n_output, n_input, n_output);
      if(ret)
        return false;
    }
  }
  for(b = 0; b < n_batch; b++)
  {
    ret = xa_nn_matXvec_f32xf32_f32(
        forget_gate_scratch+b*n_cell,
        GetBuffer<float>(input_to_forget_weights_), GetBuffer<float>(recurrent_to_forget_weights_),
        GetBuffer<float>(input_)+b*n_input, GetBuffer<float>(output_state_in_)+b*n_output,
        GetBuffer<float>(forget_gate_bias_), n_cell, n_input, n_output, n_input, n_output);
    if(ret)
      return false;
  }
  for(b = 0; b < n_batch; b++)
  {
    ret = xa_nn_matXvec_f32xf32_f32(
        cell_scratch+b*n_cell,
        GetBuffer<float>(input_to_cell_weights_), GetBuffer<float>(recurrent_to_cell_weights_), 
        GetBuffer<float>(input_)+b*n_input, GetBuffer<float>(output_state_in_)+b*n_output,
        GetBuffer<float>(cell_bias_), n_cell, n_input, n_output, n_input, n_output);
    if(ret)
      return false;
  }
  for(b = 0; b < n_batch; b++)
  {
    ret = xa_nn_matXvec_f32xf32_f32(
        output_gate_scratch+b*n_cell,
        GetBuffer<float>(input_to_output_weights_), GetBuffer<float>(recurrent_to_output_weights_),
        GetBuffer<float>(input_)+b*n_input, GetBuffer<float>(output_state_in_)+b*n_output,
        GetBuffer<float>(output_gate_bias_), n_cell, n_input, n_output, n_input, n_output);
    if(ret)
      return false;
  }

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole) {
      tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          GetBuffer<float>(cell_to_input_weights_), n_cell,
          GetBuffer<float>(cell_state_in_), n_batch, input_gate_scratch);
    }
    ret = xa_nn_vec_sigmoid_f32_f32(input_gate_scratch, input_gate_scratch, n_cell * n_batch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole) {
    for(b = 0; b < n_batch; b++)
    {
      ret = xa_nn_elm_mul_acc_f32xf32_f32(forget_gate_scratch + b * n_cell, 
          GetBuffer<float>(cell_to_forget_weights_),
          GetBuffer<float>(cell_state_in_) + b * n_cell, n_cell);
      if(ret)
        return false;
    }
  }
  ret = xa_nn_vec_sigmoid_f32_f32(forget_gate_scratch, forget_gate_scratch, n_cell * n_batch);

  // For each batch and cell: update the cell.
  ret = xa_nn_elm_mul_f32xf32_f32(GetBuffer<float>(cell_state_out_), forget_gate_scratch, 
      GetBuffer<float>(cell_state_in_), n_batch * n_cell);
#ifndef HIFI_BUILD
  switch(params_.activation_) {
#else
  switch(static_cast<TfLiteFusedActivation>(params_.activation_)) {
#endif //HiFi_BUILD
    case kTfLiteActNone:
      ret = 0;
      break;
    case kTfLiteActRelu:
      ret = xa_nn_vec_activation_min_max_f32_f32(cell_scratch, cell_scratch,
          0.0f, std::numeric_limits<float>::max(), n_batch * n_cell);
      break;
    case kTfLiteActRelu6:
      ret = xa_nn_vec_activation_min_max_f32_f32(cell_scratch, cell_scratch,
          0.0f, 6.0f, n_batch * n_cell);
      break;
    case kTfLiteActTanh:
      ret = xa_nn_vec_tanh_f32_f32(cell_scratch, cell_scratch,
          n_batch * n_cell);
      break;
    case kTfLiteActSigmoid:
      ret = xa_nn_vec_sigmoid_f32_f32(cell_scratch, cell_scratch,
          n_batch * n_cell);
      break;
    default:
      ret = -1;
  }
  if(ret != 0)
    return false;

  if (use_cifg) {
    tflite::tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                                     forget_gate_scratch);
    ret = xa_nn_elm_mul_acc_f32xf32_f32(GetBuffer<float>(cell_state_out_),
        cell_scratch, forget_gate_scratch , n_batch * n_cell);
  } else {
    ret = xa_nn_elm_mul_acc_f32xf32_f32(GetBuffer<float>(cell_state_out_),
        cell_scratch, input_gate_scratch , n_batch * n_cell);
  }
  if (params_.cell_clip_ > 0.0) {
    ret = xa_nn_vec_activation_min_max_f32_f32(GetBuffer<float>(cell_state_out_),
        GetBuffer<float>(cell_state_out_), -params_.cell_clip_, params_.cell_clip_, n_batch * n_cell);
  }

  // For each batch and cell: update the output gate.
  if (use_peephole) {
    for(b = 0; b < n_batch; b++)
    {
      ret = xa_nn_elm_mul_acc_f32xf32_f32(output_gate_scratch + b * n_cell, 
          GetBuffer<float>(cell_to_output_weights_),
          GetBuffer<float>(cell_state_out_) + b * n_cell, n_cell);
      if(ret)
        return false;
    }
  }
  ret = xa_nn_vec_sigmoid_f32_f32(output_gate_scratch, output_gate_scratch, n_batch * n_cell);

#ifndef HIFI_BUILD
  switch(params_.activation_) {
#else
  switch(static_cast<TfLiteFusedActivation>(params_.activation_)) {
#endif //HiFi_BUILD
    case kTfLiteActNone:
      ret = 0;
      break;
    case kTfLiteActRelu:
      ret = xa_nn_vec_activation_min_max_f32_f32(cell_scratch, GetBuffer<float>(cell_state_out_),
          0.0f, std::numeric_limits<float>::max(), n_batch * n_cell);
      break;
    case kTfLiteActRelu6:
      ret = xa_nn_vec_activation_min_max_f32_f32(cell_scratch, GetBuffer<float>(cell_state_out_),
          0.0f, 6.0f, n_batch * n_cell);
      break;
    case kTfLiteActTanh:
      ret = xa_nn_vec_tanh_f32_f32(cell_scratch, GetBuffer<float>(cell_state_out_),
          n_batch * n_cell);
      break;
    case kTfLiteActSigmoid:
      ret = xa_nn_vec_sigmoid_f32_f32(cell_scratch, GetBuffer<float>(cell_state_out_),
          n_batch * n_cell);
      break;
    default:
      ret = -1;
  }
  if(ret != 0)
    return false;

  ret = xa_nn_elm_mul_f32xf32_f32(output_gate_scratch, output_gate_scratch, cell_scratch, n_batch * n_cell);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight =
          (projection_weights_->lifetime != OperandLifeTime::NO_VALUE);
  const bool use_projection_bias = (projection_bias_->lifetime != OperandLifeTime::NO_VALUE);
  if (use_projection_weight) {
    if (use_projection_bias) {
      for(b = 0; b < n_batch; b++)
      {
        ret = xa_nn_matXvec_f32xf32_f32(GetBuffer<float>(output_)+b*n_output,
            GetBuffer<float>(projection_weights_), NULL,
            output_gate_scratch+b*n_cell, NULL, GetBuffer<float>(projection_bias_),
            n_output, n_cell, 0, n_cell, 0);
      }
    } else {
      tflite::tensor_utils::ZeroVector(GetBuffer<float>(output_), n_batch * n_output);
      for(b = 0; b < n_batch; b++)
      {
        ret = xa_nn_matXvec_f32xf32_f32(GetBuffer<float>(output_)+b*n_output,
            GetBuffer<float>(projection_weights_), NULL,
            output_gate_scratch+b*n_cell, NULL, GetBuffer<float>(output_)+b*n_output,
            n_output, n_cell, 0, n_cell, 0);
      }
    }
    if (params_.proj_clip_ > 0.0) {
      ret = xa_nn_vec_activation_min_max_f32_f32(GetBuffer<float>(output_),
          GetBuffer<float>(output_), -params_.cell_clip_, params_.cell_clip_, n_batch * n_cell);
    }
  } else {
    tflite::tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                             GetBuffer<float>(output_));
  }
  tflite::tensor_utils::CopyVector(GetBuffer<float>(output_), n_batch * n_output,
                           GetBuffer<float>(output_state_out_));
#endif

  return true;
}

}  // namespace nn
}  // namespace android
